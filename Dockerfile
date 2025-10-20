FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    PATH="/app/.local/bin:${PATH}" \
   # CUDA_VISIBLE_DEVICES=-1  # fallback to CPU if needed

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY pyproject.toml ./pyproject.toml
COPY uv.lock ./uv.lock

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY README.md ./README.md
COPY domaindata ./domaindata
COPY docker-entrypoint.sh ./docker-entrypoint.sh

# Storage/log folders will be created at runtime; you can mount volumes if needed
RUN mkdir -p storage logs \
    && chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["serve"]
