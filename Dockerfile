# -------------------------
# Base Image
# -------------------------
FROM python:3.11-slim

# -------------------------
# Environment
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

# -------------------------
# System Dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working Directory
# -------------------------
WORKDIR /app

# -------------------------
# Install Python Dependencies
# -------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy Project
# -------------------------
COPY . .

# -------------------------
# Create Persistent Folders
# -------------------------
RUN mkdir -p uploads .cache/huggingface
RUN mkdir -p data/chroma_db
RUN mkdir -p data/state
RUN mkdir -p data/raw
RUN mkdir -p data/processed

# -------------------------
# Expose Port
# -------------------------
EXPOSE 8000

# -------------------------
# Health check (readiness)
# -------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

# -------------------------
# Start FastAPI
# (single worker: the BM25 index lives in process memory)
# -------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]