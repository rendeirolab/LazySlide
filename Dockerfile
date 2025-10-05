# Multi-stage Dockerfile for LazySlide
# Build: docker build -t lazyslide .
# CLI usage: docker run -it --rm lazyslide python
# Jupyter usage: docker run -it --rm -p 8888:8888 lazyslide jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# IPython usage: docker run -it --rm lazyslide ipython

# Use a smaller base image for runtime
FROM python:3.13-slim AS base

# Set environment variables for faster builds and better performance
ENV PYTHONUNBUFFERED=1
ENV CFLAGS="-O2 -march=native"
ENV CXXFLAGS="-O2 -march=native"
ENV MAX_JOBS=4
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV HF_HOME=/tmp/hf_home
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv for faster Python package management
RUN pip install --no-cache-dir uv

# Dependencies stage - install Python packages
FROM base AS deps
WORKDIR /app

# Copy git directory first for version detection
COPY .git/ ./.git/

# Copy dependency files and README (required for package metadata)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies without the project
RUN uv sync --group tests --group dev --extra all --extra models --no-install-project

# Install Jupyter Lab and IPython for interactive use
RUN uv pip install jupyterlab ipython  numcodecs

# Build stage - install the project
FROM deps AS builder
WORKDIR /app

# Copy source code, tests, and README
COPY src/ ./src/
COPY tests/ ./tests/
COPY README.md ./

# Install the project in development mode
RUN uv run pip install -e .

# Runtime stage - minimal image
FROM python:3.13-slim AS runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV HF_HOME=/tmp/hf_home
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy only necessary files
COPY --from=builder /app/src/ ./src/
COPY --from=builder /app/tests/ ./tests/
COPY pyproject.toml ./

# Verify installation
RUN python -c "import lazyslide as zs; print(f'LazySlide version: {zs.__version__}')"

# Default command - provide a shell for manual execution
CMD ["/bin/bash"]
