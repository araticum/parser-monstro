"""
Parser Monstro - Serviço de extração de texto de documentos de licitação.
API REST FastAPI — roda em container isolado na porta 7000.
"""

import asyncio
import io
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import magic
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
PARSER_MODE = os.getenv("PARSER_MODE", "balanced").strip().lower()
ENABLE_EASYOCR = os.getenv("ENABLE_EASYOCR", "false").lower() == "true"
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1" if PARSER_MODE == "precision_first" else "2"))
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "/app/storage"))

# Precision-first knobs (safe defaults for quality)
FORCE_OCR_IF_SCORE_BELOW = float(os.getenv("FORCE_OCR_IF_SCORE_BELOW", "0.82" if PARSER_MODE == "precision_first" else "0.65"))
REPROCESS_IF_SCORE_BELOW = float(os.getenv("REPROCESS_IF_SCORE_BELOW", "0.72" if PARSER_MODE == "precision_first" else "0.55"))
MIN_CHARS_PER_PAGE_NATIVE = int(os.getenv("MIN_CHARS_PER_PAGE_NATIVE", "180" if PARSER_MODE == "precision_first" else "80"))
CLEAN_OCR_NOISE = os.getenv("CLEAN_OCR_NOISE", "true").lower() == "true"

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("parser-monstro")

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
jobs: Dict[str, Dict[str, Any]] = {}
semaphore: asyncio.Semaphore  # initialised in lifespan


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class DocumentInput(BaseModel):
    url: str
    filename: str


class ParseOptions(BaseModel):
    enable_easyocr: bool = False
    force_ocr: bool = False


class ParseRequest(BaseModel):
    tender_id: str
    documents: List[DocumentInput]
    options: Optional[ParseOptions] = None


class DocumentResult(BaseModel):
    filename: str
    type_detected: str
    method_used: str
    pages: int
    quality_score: float
    text: str
    error: Optional[str] = None


class ParseResponse(BaseModel):
    tender_id: str
    status: str  # done | error | pending | processing
    job_id: str
    documents: List[DocumentResult] = []
    full_text: str = ""
    errors: List[str] = []
    processing_time_s: float = 0.0


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Parser Monstro", version="1.0.0")


@app.on_event("startup")
async def startup():
    global semaphore
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Parser Monstro iniciado. mode=%s MAX_WORKERS=%d EASYOCR=%s OCR<%.2f REPROCESS<%.2f",
        PARSER_MODE,
        MAX_WORKERS,
        ENABLE_EASYOCR,
        FORCE_OCR_IF_SCORE_BELOW,
        REPROCESS_IF_SCORE_BELOW,
    )


# ---------------------------------------------------------------------------
# Health & queue
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        tess_ok = True
    except Exception:
        tess_ok = False
    return {
        "status": "ok",
        "tesseract": tess_ok,
        "easyocr_enabled": ENABLE_EASYOCR,
        "parser_mode": PARSER_MODE,
        "force_ocr_if_score_below": FORCE_OCR_IF_SCORE_BELOW,
        "reprocess_if_score_below": REPROCESS_IF_SCORE_BELOW,
    }


@app.get("/queue")
def queue_status():
    pending = sum(1 for j in jobs.values() if j["status"] == "pending")
    processing = sum(1 for j in jobs.values() if j["status"] == "processing")
    done = sum(1 for j in jobs.values() if j["status"] in ("done", "error"))
    return {"pending": pending, "processing": processing, "done": done}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


# ---------------------------------------------------------------------------
# Main parse endpoint
# ---------------------------------------------------------------------------
@app.post("/parse", response_model=ParseResponse, status_code=202)
async def parse_documents(req: ParseRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "tender_id": req.tender_id,
        "status": "pending",
        "documents": [],
        "full_text": "",
        "errors": [],
        "processing_time_s": 0.0,
        "created_at": time.time(),
    }
    background_tasks.add_task(_process_job, job_id, req)
    return ParseResponse(
        tender_id=req.tender_id,
        status="pending",
        job_id=job_id,
    )


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------
async def _process_job(job_id: str, req: ParseRequest):
    async with semaphore:
        jobs[job_id]["status"] = "processing"
        t0 = time.time()
        doc_results: List[Dict] = []
        errors: List[str] = []
        use_easyocr = (req.options and req.options.enable_easyocr) or ENABLE_EASYOCR
        force_ocr = bool(req.options and req.options.force_ocr)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for doc in req.documents:
                try:
                    results = await _handle_document(doc, tmp, use_easyocr, force_ocr)
                    doc_results.extend(results)
                except Exception as exc:
                    logger.exception("Erro ao processar %s", doc.filename)
                    errors.append(f"{doc.filename}: {exc}")

        full_text = "\n\n".join(r["text"] for r in doc_results if r.get("text"))
        jobs[job_id].update(
            status="done" if not errors or doc_results else "error",
            documents=doc_results,
            full_text=full_text,
            errors=errors,
            processing_time_s=round(time.time() - t0, 2),
        )
        logger.info("Job %s concluído em %.1fs (%d docs)", job_id, jobs[job_id]["processing_time_s"], len(doc_results))


async def _handle_document(doc: DocumentInput, tmpdir: Path, use_easyocr: bool, force_ocr: bool) -> List[Dict]:
    """Download + decompress + parse one document. Returns list of DocumentResult dicts."""
    # Download
    dest = tmpdir / doc.filename
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(doc.url)
        resp.raise_for_status()
    dest.write_bytes(resp.content)

    # Detect real mime type
    mime = magic.from_file(str(dest), mime=True)
    logger.debug("%s → mime=%s", doc.filename, mime)

    # Decompress archives
    files_to_parse: List[Path] = []
    if mime in ("application/zip", "application/x-zip-compressed") or doc.filename.lower().endswith(".zip"):
        files_to_parse = _extract_zip(dest, tmpdir)
    elif mime in ("application/x-rar-compressed", "application/vnd.rar") or doc.filename.lower().endswith(".rar"):
        files_to_parse = _extract_rar(dest, tmpdir)
    elif mime == "application/x-7z-compressed" or doc.filename.lower().endswith(".7z"):
        files_to_parse = _extract_7z(dest, tmpdir)
    else:
        files_to_parse = [dest]

    results = []
    for fpath in files_to_parse:
        result = _parse_file(fpath, use_easyocr, force_ocr)
        results.append(result)
    return results


def _extract_zip(path: Path, dest: Path) -> List[Path]:
    out = dest / f"_zip_{path.stem}"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out)
    return [p for p in out.rglob("*") if p.is_file()]


def _extract_rar(path: Path, dest: Path) -> List[Path]:
    import rarfile
    out = dest / f"_rar_{path.stem}"
    out.mkdir(exist_ok=True)
    with rarfile.RarFile(str(path)) as rf:
        rf.extractall(str(out))
    return [p for p in out.rglob("*") if p.is_file()]


def _extract_7z(path: Path, dest: Path) -> List[Path]:
    import py7zr
    out = dest / f"_7z_{path.stem}"
    out.mkdir(exist_ok=True)
    with py7zr.SevenZipFile(str(path), mode="r") as zf:
        zf.extractall(path=str(out))
    return [p for p in out.rglob("*") if p.is_file()]


def _parse_file(path: Path, use_easyocr: bool, force_ocr: bool) -> Dict:
    """Parse a single file with fallback chain."""
    mime = magic.from_file(str(path), mime=True)
    filename = path.name

    # Route by type
    if "pdf" in mime:
        return _parse_pdf(path, use_easyocr, force_ocr)
    elif "word" in mime or "officedocument" in mime or path.suffix.lower() in (".doc", ".docx"):
        return _parse_docx(path)
    elif "html" in mime or path.suffix.lower() in (".html", ".htm"):
        return _parse_html(path)
    elif "text" in mime or path.suffix.lower() == ".txt":
        text = path.read_text(errors="replace")
        return _make_result(filename, "plain_text", "read", 1, 1.0, text)
    elif mime.startswith("image/"):
        return _parse_image_ocr(path, use_easyocr)
    else:
        return _make_result(filename, mime, "unsupported", 0, 0.0, "", error=f"Tipo não suportado: {mime}")


def _parse_pdf(path: Path, use_easyocr: bool, force_ocr: bool) -> Dict:
    filename = path.name
    text = ""
    method = ""
    pages = 0

    # 1) pdfplumber (preferência para texto nativo)
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = len(pdf.pages)
            parts = [(pg.extract_text() or "") for pg in pdf.pages]
            text = "\n".join(parts).strip()
            if text:
                method = "pdfplumber"
    except Exception as e:
        logger.debug("pdfplumber falhou em %s: %s", filename, e)

    # 2) pymupdf fallback para texto nativo ruim/incompleto
    if not text:
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(path))
            pages = doc.page_count
            parts = [doc.load_page(i).get_text() for i in range(pages)]
            text = "\n".join(parts).strip()
            doc.close()
            if text:
                method = "pymupdf"
        except Exception as e:
            logger.debug("pymupdf falhou em %s: %s", filename, e)

    text = _normalize_text(text)
    quality = _quality_score(text, pages)

    chars_per_page = (len(text) / max(1, pages)) if text else 0
    native_is_weak = quality < FORCE_OCR_IF_SCORE_BELOW or chars_per_page < MIN_CHARS_PER_PAGE_NATIVE

    # 3) OCR obrigatório quando forçado ou quando extração nativa é fraca
    if force_ocr or not text or native_is_weak:
        try:
            ocr_text, pages_ocr = _pdf_ocr_tesseract(path, use_easyocr)
            ocr_text = _normalize_text(ocr_text)
            ocr_pages = pages_ocr or pages
            ocr_quality = _quality_score(ocr_text, ocr_pages)

            if force_ocr or ocr_quality >= quality or quality < REPROCESS_IF_SCORE_BELOW:
                text = ocr_text
                pages = ocr_pages
                quality = ocr_quality
                method = "easyocr" if use_easyocr else "tesseract"
        except Exception as e:
            logger.debug("tesseract OCR falhou em %s: %s", filename, e)

    type_detected = "pdf_native" if method in ("pdfplumber", "pymupdf") else "pdf_scanned"
    return _make_result(filename, type_detected, method or "failed", pages, quality, text)


def _pdf_ocr_tesseract(path: Path, use_easyocr: bool) -> tuple:
    """Render PDF pages as images and OCR them."""
    import pytesseract
    from PIL import Image
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), dpi=200)
    except Exception:
        # fallback: use pymupdf to render
        import fitz
        doc = fitz.open(str(path))
        images = []
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()

    parts = []
    for img in images:
        if use_easyocr:
            try:
                import easyocr
                reader = easyocr.Reader(["pt", "en"])
                import numpy as np
                result = reader.readtext(np.array(img))
                parts.append(" ".join(r[1] for r in result))
                continue
            except Exception:
                pass
        t = pytesseract.image_to_string(img, lang="por+eng")
        parts.append(t)

    return "\n".join(parts).strip(), len(images)


def _parse_docx(path: Path) -> Dict:
    filename = path.name
    try:
        from docx import Document
        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        pages = max(1, len(text) // 3000)
        return _make_result(filename, "docx", "python-docx", pages, _quality_score(text, pages), text)
    except Exception as e:
        return _make_result(filename, "docx", "failed", 0, 0.0, "", error=str(e))


def _parse_html(path: Path) -> Dict:
    filename = path.name
    try:
        from bs4 import BeautifulSoup
        html = path.read_text(errors="replace")
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n")
        return _make_result(filename, "html", "beautifulsoup4", 1, _quality_score(text, 1), text)
    except Exception as e:
        return _make_result(filename, "html", "failed", 0, 0.0, "", error=str(e))


def _parse_image_ocr(path: Path, use_easyocr: bool) -> Dict:
    filename = path.name
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(str(path))
        if use_easyocr:
            try:
                import easyocr
                import numpy as np
                reader = easyocr.Reader(["pt", "en"])
                result = reader.readtext(np.array(img))
                text = " ".join(r[1] for r in result)
                return _make_result(filename, "image", "easyocr", 1, _quality_score(text, 1), text)
            except Exception:
                pass
        text = pytesseract.image_to_string(img, lang="por+eng")
        return _make_result(filename, "image", "tesseract", 1, _quality_score(text, 1), text)
    except Exception as e:
        return _make_result(filename, "image", "failed", 0, 0.0, "", error=str(e))


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    txt = unicodedata.normalize("NFKC", text)
    txt = txt.replace("\x00", " ")

    if CLEAN_OCR_NOISE:
        # remove linhas com ruído típico de OCR (quase sem vogais e muito símbolo)
        cleaned_lines = []
        for line in txt.splitlines():
            s = line.strip()
            if not s:
                continue
            non_word_ratio = len(re.findall(r"[^\w\s]", s)) / max(1, len(s))
            vowels = len(re.findall(r"[aeiouáéíóúâêôãõàü]", s, flags=re.IGNORECASE))
            if len(s) >= 24 and vowels == 0 and non_word_ratio > 0.25:
                continue
            cleaned_lines.append(s)
        txt = "\n".join(cleaned_lines)

    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _quality_score(text: str, pages: int) -> float:
    if not text or not pages:
        return 0.0
    words = len(text.split())
    words_per_page = words / max(pages, 1)
    # Heuristic: ~250 words/page = good quality (1.0)
    score = min(1.0, words_per_page / 250)
    return round(score, 2)


def _make_result(filename, type_detected, method, pages, quality, text, error=None) -> Dict:
    normalized_text = _normalize_text(text)
    quality = _quality_score(normalized_text, pages)
    return {
        "filename": filename,
        "type_detected": type_detected,
        "method_used": method,
        "pages": pages,
        "quality_score": quality,
        "text": normalized_text,
        "error": error,
    }
