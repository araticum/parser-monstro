FROM python:3.11

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-por \
    libmagic1 \
    poppler-utils \
    p7zip-full \
    unar \
    libreoffice-writer \
    libreoffice-calc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Core deps (obrigatório)
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Heavy deps: docling, marker-pdf (opcional — falha não quebra container)
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt || echo "WARNING: heavy deps failed, continuing without them"
# Install torch with ROCm support (AMD Radeon) instead of default CUDA
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3


# Opt-in EasyOCR (heavy, GPU-optional)
ARG ENABLE_EASYOCR=false
RUN if [ "$ENABLE_EASYOCR" = "true" ]; then \
    pip install --no-cache-dir easyocr torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi

COPY main.py .

RUN mkdir -p /app/storage

EXPOSE 7000

ENV ENABLE_EASYOCR=false
ENV MAX_WORKERS=2
ENV LOG_LEVEL=INFO

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "1"]
