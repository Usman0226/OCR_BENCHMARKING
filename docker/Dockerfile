# =============================================================================
# OCR Benchmark — Dockerfile
# Base: Ubuntu 24.04 LTS | Python 3.11 | CPU-only
# PaddlePaddle CPU + PaddleOCR + Tesseract 5 (eng + fra)
# Non-root user | BuildKit cache mounts | Optimized layers
# Build context: project root (ocr-position-benchmark/)
# =============================================================================

# syntax=docker/dockerfile:1.6
FROM ubuntu:24.04 AS base

# Avoid interactive prompts during apt operations
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =============================================================================
# Stage 1: System packages + Tesseract
# =============================================================================
FROM base AS system

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        # Python 3.11
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        # Build tools
        build-essential \
        gcc \
        g++ \
        cmake \
        # Tesseract OCR 5 + language packs (English + French)
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
        # OpenCV system dependencies
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        # Image libraries
        libjpeg-turbo8 \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        # Utilities
        wget \
        curl \
        git \
        ca-certificates \
        locales \
        && locale-gen en_US.UTF-8 \
        && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
        && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
        && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
        && python3.11 -m pip install --upgrade pip setuptools wheel \
        && rm -rf /tmp/* /root/.cache

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# =============================================================================
# Stage 2: Python dependencies
# PaddlePaddle CPU must be installed before PaddleOCR
# =============================================================================
FROM system AS python-deps

WORKDIR /build

# Copy only dependency files first (maximises layer cache hits)
COPY requirements.txt pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    # Install PaddlePaddle CPU first (large package, needs separate index)
    pip install --no-cache-dir --ignore-installed \
        paddlepaddle==2.6.2 \
        -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
    && pip install --no-cache-dir --ignore-installed -r requirements.txt

# =============================================================================
# Stage 3: Final runtime image
# =============================================================================
FROM system AS runtime

# Copy installed Python packages from build stage
COPY --from=python-deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Create non-root user (UID 1000). Ubuntu 24.04 uses 1000 for the 'ubuntu' user, so remove it first.
RUN userdel -r ubuntu || true
RUN groupadd --gid 1000 ocr \
    && useradd --uid 1000 --gid ocr --shell /bin/bash --create-home ocr

# Create runtime directories owned by the non-root user
# /workspace is bind-mounted at runtime — just ensure the mount point exists
RUN mkdir -p \
        /workspace \
        /paddle-models \
        /logs \
    && chown -R ocr:ocr \
        /workspace \
        /paddle-models \
        /logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace
ENV PADDLE_MODEL_DIR=/paddle-models
ENV OCR_LOG_DIR=/logs
# results.csv and results_summary.md written into /workspace (project root)
ENV OCR_RESULTS_DIR=/workspace
ENV OCR_REPORTS_DIR=/workspace/reports
ENV OCR_OUTPUT_DIR=/workspace/ocr_output
ENV CONFIG_PATH=/workspace/configs/config.yaml

# Tesseract tessdata path (Ubuntu 24.04 Tesseract 5)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

WORKDIR /workspace

# Switch to non-root user
USER ocr

# Default: keep container alive for docker compose exec / Dev Container
CMD ["sleep", "infinity"]
