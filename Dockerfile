# ─────────────────────────────────────────────────────────────────
# Epidemiological Pulse – Production Dockerfile
# Multi-stage build: builder → runtime
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

LABEL maintainer="epidemiological-pulse"
LABEL description="Population Health Hotspot Detection System"
LABEL version="1.0.0"

# System dependencies required for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Use a virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools first
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python dependencies (leverages Docker layer cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libraries only (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── App setup ─────────────────────────────────────────────────────
WORKDIR /app

# Create non-root user for security
RUN groupadd -r epulse && useradd -r -g epulse -s /bin/false epulse

# Copy project files
COPY config/       ./config/
COPY data/         ./data/
COPY src/          ./src/
COPY outputs/      ./outputs/
COPY logs/         ./logs/

# Ensure output/log directories are writable
RUN mkdir -p logs outputs/geojson outputs/models outputs/reports \
    && chown -R epulse:epulse /app

# Switch to non-root user
USER epulse

# ── Environment variables ─────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CONFIG_PATH=/app/config/params.yaml \
    API_KEYS_PATH=/app/config/api_keys.yaml \
    DASH_HOST=0.0.0.0 \
    DASH_PORT=8050 \
    DASH_DEBUG=false

# Expose dashboard port
EXPOSE 8050

# ── Health check ──────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# ── Entry point ───────────────────────────────────────────────────
# Default: run the full pipeline then launch the dashboard
# Override with: docker run ... python src/pipeline.py --mode <mode>
ENTRYPOINT ["python", "src/pipeline.py"]
CMD ["--mode", "full", "--generate"]
