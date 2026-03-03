FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    tesseract-ocr \
    tesseract-ocr-por \
    libmagic1 \
    poppler-utils \
    unrar \
    p7zip-full \
    libreoffice \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
