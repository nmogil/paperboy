# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.prod.txt ./
RUN pip install --no-cache-dir -r requirements.prod.txt

# Install Playwright in builder stage
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 && \
    pip install --no-cache-dir playwright && \
    # Install Playwright with specific user
    mkdir -p $PLAYWRIGHT_BROWSERS_PATH && \
    chown -R root:root $PLAYWRIGHT_BROWSERS_PATH && \
    chmod -R 755 $PLAYWRIGHT_BROWSERS_PATH && \
    playwright install chromium --with-deps && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
    /root/.cache/ms-playwright \
    /root/.cache/pip && \
    # Remove other browsers and set strict permissions
    find $PLAYWRIGHT_BROWSERS_PATH -mindepth 1 -maxdepth 1 ! -name "chromium-*" -exec rm -rf {} + && \
    chmod -R 755 $PLAYWRIGHT_BROWSERS_PATH/chromium-*

# Final stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright

# Set working directory
WORKDIR /app

# Copy only the necessary files from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder $PLAYWRIGHT_BROWSERS_PATH $PLAYWRIGHT_BROWSERS_PATH

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with specific UID/GID
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --no-create-home app && \
    mkdir -p /app/config && \
    # Set strict permissions
    chown -R app:app /app && \
    chmod -R 750 /app && \
    chmod 440 /app/config && \
    # Ensure Playwright permissions
    chown -R app:app $PLAYWRIGHT_BROWSERS_PATH && \
    chmod -R 750 $PLAYWRIGHT_BROWSERS_PATH

# Copy application files with strict permissions
COPY --chown=app:app src/ /app/src/
RUN chmod -R 550 /app/src  # Read and execute only

# Mount point for config - will be handled by docker-compose
RUN mkdir -p /app/config && \
    chown app:app /app/config && \
    chmod 750 /app/config

# Set HOME and switch to non-root user
ENV HOME=/tmp
USER 10001:10001

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--log-level", "warning"] 