# LandPPT Docker Image
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- MODIFICATION v2 START ---
# The python:3.11-slim image (Debian Bookworm) uses a new source file format.
# We must remove the default source file before adding our own to prevent conflicts.
RUN rm /etc/apt/sources.list.d/debian.sources && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main" >> /etc/apt/sources.list
# --- MODIFICATION v2 END ---

# Install system dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN pip install uv

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv pip install --system apryse-sdk>=11.5.0 --extra-index-url=https://pypi.apryse.com
RUN uv pip install --system -r pyproject.toml

# Production stage
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PYPPETEER_CHROMIUM_REVISION=1263111 \
    PYPPETEER_EXECUTABLE_PATH=/usr/bin/chromium

# --- MODIFICATION v2 START ---
# Also remove the default source file in the production stage.
RUN rm /etc/apt/sources.list.d/debian.sources && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main" >> /etc/apt/sources.list
# --- MODIFICATION v2 END ---

# Install system dependencies for runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # For PDF processing
    wkhtmltopdf \
    # For file processing
    poppler-utils \
    # For image processing
    libmagic1 \
    # For network requests
    ca-certificates \
    curl \
    # For pyppeteer (Chromium dependencies)
    chromium \
    chromium-driver \
    # For onnxruntime
    libgomp1 \
    # For general compatibility
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    # For netcat (health check)
    netcat-openbsd \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r landppt && useradd -r -g landppt landppt

# Set work directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY run.py setup_database.py ./
COPY template_examples/ ./template_examples/
COPY docker-healthcheck.sh docker-entrypoint.sh ./

# Make scripts executable
RUN chmod +x docker-healthcheck.sh docker-entrypoint.sh

# Create necessary directories and set ownership
RUN mkdir -p temp/ai_responses_cache \
    temp/style_genes_cache \
    temp/summeryanyfile_cache \
    temp/templates_cache \
    research_reports \
    lib/Linux \
    lib/MacOS \
    lib/Windows \
    uploads \
    && chown -R landppt:landppt /app

# Switch to non-root user
USER landppt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD ./docker-healthcheck.sh

# Set entrypoint and default command
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "run.py"]