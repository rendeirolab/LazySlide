# Multi-stage Dockerfile for LazySlide
# Build: DOCKER_BUILDKIT=1 docker build -t lazyslide .
# CLI usage: docker run -it --rm lazyslide python
# Jupyter usage: docker run -it --rm -p 8888:8888 lazyslide jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# IPython usage: docker run -it --rm lazyslide ipython

# Use a smaller base image for runtime
FROM python:3.13-slim AS builder
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables for faster builds and better performance
ENV PYTHONUNBUFFERED=1 \
    MAX_JOBS=4 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    HF_HOME=/tmp/hf_home \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/default-java \
    PATH="/usr/lib/jvm/default-java/bin:$PATH"

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y \
    default-jdk \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy git directory first for version detection
COPY .git/ ./.git/

# Copy dependency files and README (required for package metadata)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies without the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra all --extra models --no-install-project

# Install Jupyter Lab and IPython for interactive use
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install jupyterlab ipython \
    torchstain scanpy spatialdata-plot scyjava cucim hf-xet \
    igraph ipywidgets marsilea mpl-fontkit

# Copy source and build
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel && uv pip install dist/*.whl

# Runtime stage - minimal image
FROM python:3.13-slim AS runtime
# Include uv in case user wants to use it
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    HF_HOME=/tmp/hf_home \
    JAVA_HOME=/usr/lib/jvm/default-java \
    PATH="/app/.venv/bin:/usr/lib/jvm/default-java/bin:$PATH"

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Verify installation
RUN python -c "import lazyslide as zs; import scyjava; import cucim; \
    print(f'LazySlide version: {zs.__version__}')"

# Default command - provide a shell for manual execution
CMD ["/bin/bash"]
