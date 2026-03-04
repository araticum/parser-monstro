"""
Parser Monstro - Serviço de extração de texto de documentos de licitação.
API REST FastAPI — roda em container isolado na porta 7000.
"""

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
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
purge_tasks: Dict[str, asyncio.Task] = {}
purge_index_path = STORAGE_ROOT / ".purge_index.json"
purge_index_lock = asyncio.Lock()


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
    purge_after_days: int = 7
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


class EnrichmentResult(BaseModel):
    tender_id: str
    chunks_total: int
    chunks_ok: int
    processing_time_s: float
    resumo_ia: Optional[str] = None
    regras_licitacao: Optional[dict] = None
    itens: Optional[list] = None
    fornecedores_sugeridos: Optional[dict] = None
    raw_chunks: Optional[list] = None
    created_at: str


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Parser Monstro", version="1.0.0")


@app.on_event("startup")
async def startup():
    global semaphore
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    await _restore_and_schedule_purges()
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


@app.get("/storage/{tender_id}")
def list_storage(tender_id: str):
    target_dir = STORAGE_ROOT / tender_id
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail="Storage not found for tender")
    files = [str(p.relative_to(target_dir)) for p in target_dir.rglob("*") if p.is_file()]
    return {
        "tender_id": tender_id,
        "storage_path": str(target_dir),
        "files": sorted(files),
        "count": len(files),
    }


@app.post("/storage/{tender_id}/enrichment")
def save_enrichment(tender_id: str, payload: EnrichmentResult):
    if payload.tender_id != tender_id:
        raise HTTPException(status_code=400, detail="tender_id mismatch between path and payload")

    target_dir = STORAGE_ROOT / tender_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "enrichment.json"
    target_file.write_text(
        json.dumps(payload.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "tender_id": tender_id,
        "saved_path": str(target_file),
    }


@app.get("/storage/{tender_id}/enrichment")
def get_enrichment(tender_id: str):
    target_file = STORAGE_ROOT / tender_id / "enrichment.json"
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="Enrichment not found for tender")

    try:
        return json.loads(target_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read enrichment file: {exc}")


@app.delete("/storage/{tender_id}")
async def delete_storage(tender_id: str):
    deleted = await _purge_tender_storage(tender_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Storage not found for tender")
    return {"tender_id": tender_id, "deleted": True}


# ---------------------------------------------------------------------------
# Main parse endpoint
# ---------------------------------------------------------------------------
@app.post("/parse", response_model=ParseResponse, status_code=202)
async def parse_documents(req: ParseRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    purge_at = datetime.now(timezone.utc) + timedelta(days=max(1, req.purge_after_days))
    storage_path = STORAGE_ROOT / req.tender_id
    jobs[job_id] = {
        "job_id": job_id,
        "tender_id": req.tender_id,
        "status": "pending",
        "documents": [],
        "full_text": "",
        "errors": [],
        "processing_time_s": 0.0,
        "created_at": time.time(),
        "storage_path": str(storage_path),
        "purge_at": purge_at.isoformat(),
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
        target_dir = STORAGE_ROOT / req.tender_id
        target_dir.mkdir(parents=True, exist_ok=True)

        for doc in req.documents:
            try:
                results = await _handle_document(doc, target_dir, use_easyocr, force_ocr)
                doc_results.extend(results)
            except Exception as exc:
                logger.exception("Erro ao processar %s", doc.filename)
                errors.append(f"{doc.filename}: {exc}")

        full_text = "\n\n".join(r["text"] for r in doc_results if r.get("text"))
        jobs[job_id].update(
            status="done" if doc_results else "error",
            documents=doc_results,
            full_text=full_text,
            errors=errors,
            processing_time_s=round(time.time() - t0, 2),
            storage_path=str(target_dir),
        )

        purge_at = datetime.now(timezone.utc) + timedelta(days=max(1, req.purge_after_days))
        jobs[job_id]["purge_at"] = purge_at.isoformat()
        await _upsert_purge_schedule(req.tender_id, purge_at)

        logger.info("Job %s concluído em %.1fs (%d docs)", job_id, jobs[job_id]["processing_time_s"], len(doc_results))


async def _handle_document(doc: DocumentInput, tmpdir: Path, use_easyocr: bool, force_ocr: bool) -> List[Dict]:
    """Download + decompress + parse one document. Returns list of DocumentResult dicts."""
    safe_name = Path(doc.filename).name or f"doc_{uuid.uuid4().hex}"
    dest = tmpdir / safe_name
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
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


async def _restore_and_schedule_purges() -> None:
    if not purge_index_path.exists():
        return
    try:
        raw = json.loads(purge_index_path.read_text())
    except Exception:
        logger.warning("Falha ao ler índice de purge; ignorando")
        return

    now = datetime.now(timezone.utc)
    changed = False
    for tender_id, purge_at_iso in raw.items():
        try:
            purge_at = datetime.fromisoformat(purge_at_iso)
            if purge_at.tzinfo is None:
                purge_at = purge_at.replace(tzinfo=timezone.utc)
        except Exception:
            changed = True
            continue

        if purge_at <= now:
            await _purge_tender_storage(tender_id)
            changed = True
            continue
        _schedule_purge_task(tender_id, purge_at)

    if changed:
        await _save_purge_index(await _load_purge_index())


async def _load_purge_index() -> Dict[str, str]:
    async with purge_index_lock:
        if not purge_index_path.exists():
            return {}
        try:
            return json.loads(purge_index_path.read_text())
        except Exception:
            return {}


async def _save_purge_index(data: Dict[str, str]) -> None:
    async with purge_index_lock:
        purge_index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _schedule_purge_task(tender_id: str, purge_at: datetime) -> None:
    existing = purge_tasks.get(tender_id)
    if existing and not existing.done():
        existing.cancel()
    purge_tasks[tender_id] = asyncio.create_task(_purge_after_delay(tender_id, purge_at))


async def _upsert_purge_schedule(tender_id: str, purge_at: datetime) -> None:
    data = await _load_purge_index()
    data[tender_id] = purge_at.isoformat()
    await _save_purge_index(data)
    _schedule_purge_task(tender_id, purge_at)


async def _purge_after_delay(tender_id: str, purge_at: datetime) -> None:
    delay = max(0, (purge_at - datetime.now(timezone.utc)).total_seconds())
    await asyncio.sleep(delay)
    await _purge_tender_storage(tender_id)


async def _purge_tender_storage(tender_id: str) -> bool:
    target_dir = STORAGE_ROOT / tender_id
    existed = target_dir.exists() and target_dir.is_dir()
    if existed:
        import shutil
        shutil.rmtree(target_dir, ignore_errors=True)

    data = await _load_purge_index()
    if tender_id in data:
        data.pop(tender_id, None)
        await _save_purge_index(data)

    task = purge_tasks.pop(tender_id, None)
    if task and not task.done():
        task.cancel()
    return existed


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
    """Parse PDF with ordered fallback chain:
    1. PyMuPDF  — fast native extraction using embedded text
    2. Docling  — high-quality layout-aware engine
    3. Marker   — handles difficult/scanned PDFs → Markdown output
    """
    filename = path.name
    text = ""
    method = ""
    pages = 0

    # ── 1) PyMuPDF — rápido, extração nativa ──────────────────────────────────
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = doc.page_count
        parts = [(doc.load_page(i).get_text("text") or "") for i in range(pages)]
        doc.close()
        text = "\n".join(parts).strip()
        if text:
            method = "pymupdf"
    except Exception as e:
        logger.debug("pymupdf falhou em %s: %s", filename, e)

    text = _normalize_text(text)
    quality = _quality_score(text, pages)
    chars_per_page = (len(text) / max(1, pages)) if text else 0
    native_is_weak = quality < FORCE_OCR_IF_SCORE_BELOW or chars_per_page < MIN_CHARS_PER_PAGE_NATIVE

    # ── 2) Docling — engine principal de qualidade ─────────────────────────────
    if not text or native_is_weak or force_ocr:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
            converter = DocumentConverter()
            result = converter.convert(str(path))
            docling_text = result.document.export_to_text() if result and result.document else ""
            docling_text = _normalize_text(docling_text)
            if docling_text:
                docling_pages = pages or 1
                docling_quality = _quality_score(docling_text, docling_pages)
                if force_ocr or docling_quality > quality or quality < REPROCESS_IF_SCORE_BELOW:
                    text = docling_text
                    pages = docling_pages
                    quality = docling_quality
                    method = "docling"
                    native_is_weak = quality < FORCE_OCR_IF_SCORE_BELOW
        except ImportError:
            logger.warning("docling não instalado — pulando etapa 2 do fallback para %s", filename)
        except Exception as e:
            logger.debug("docling falhou em %s: %s", filename, e)

    # ── 3) Marker — fallback para PDFs difíceis → Markdown ───────────────────
    if not text or native_is_weak:
        try:
            from marker.convert import convert_single_pdf  # type: ignore
            from marker.models import load_all_models  # type: ignore
            marker_models = load_all_models()
            full_text_md, _doc_images, _metadata = convert_single_pdf(str(path), marker_models)
            marker_text = _normalize_text(full_text_md or "")
            if marker_text:
                marker_pages = pages or 1
                marker_quality = _quality_score(marker_text, marker_pages)
                if force_ocr or marker_quality > quality or quality < REPROCESS_IF_SCORE_BELOW:
                    text = marker_text
                    pages = marker_pages
                    quality = marker_quality
                    method = "marker"
        except ImportError:
            logger.warning("marker não instalado — pulando etapa 3 do fallback para %s", filename)
        except Exception as e:
            logger.debug("marker falhou em %s: %s", filename, e)

    type_detected = "pdf_native" if method in ("pymupdf", "docling", "marker") else "pdf_scanned"
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
