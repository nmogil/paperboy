# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    PYTHONPATH=/app

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
    PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
    PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/local/ms-playwright/chromium-*/chrome \
    CHROMIUM_FLAGS="--disable-dev-shm-usage --no-sandbox --disable-setuid-sandbox --disable-gpu --disable-software-rasterizer --no-zygote" \
    PORT=8000 \
    PYTHONPATH=/app \
    # Cloud Run recommended environment variables
    GUNICORN_WORKERS=1 \
    GUNICORN_THREADS=8 \
    GUNICORN_TIMEOUT=0 \
    GUNICORN_KEEP_ALIVE=65 \
    # DBus and Chrome configuration
    DBUS_SESSION_BUS_ADDRESS="unix:path=/tmp/dbus/session_bus_socket" \
    DISPLAY=:99 \
    # Chrome stability settings
    CHROME_NO_SANDBOX=1 \
    PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/local/ms-playwright/chromium-*/chrome

# Set working directory
WORKDIR /app

# Copy only the necessary files from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder $PLAYWRIGHT_BROWSERS_PATH $PLAYWRIGHT_BROWSERS_PATH

# Install only runtime dependencies - updated with more complete Chrome dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Basic system utilities
    curl \
    iputils-ping \
    net-tools \
    xvfb \
    dbus \
    dbus-x11 \
    # Chrome dependencies
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
    libglib2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-dri3-0 \
    libxcb-shm0 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxtst6 \
    libxss1 \
    # Font support
    fonts-liberation \
    fonts-noto-color-emoji \
    # Additional dependencies
    libu2f-udev \
    xdg-utils && \
    # Create DBus machine ID and proper directory structure
    mkdir -p /var/run/dbus /var/lib/dbus /tmp/dbus && \
    dbus-uuidgen > /var/lib/dbus/machine-id && \
    chmod 644 /var/lib/dbus/machine-id && \
    # Set up DBus session for non-root user
    echo '#!/bin/bash\nexport DBUS_SESSION_BUS_ADDRESS="unix:path=/tmp/dbus/session_bus_socket"\nmkdir -p /tmp/dbus\ndbus-daemon --session --address="$DBUS_SESSION_BUS_ADDRESS" --nofork --nopidfile --syslog-only &\nexec "$@"' > /usr/local/bin/start-with-dbus.sh && \
    chmod +x /usr/local/bin/start-with-dbus.sh && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with specific UID/GID
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --no-create-home app && \
    mkdir -p /app/config /tmp/dbus && \
    # Set strict permissions
    chown -R app:app /app && \
    chmod -R 750 /app && \
    chmod 440 /app/config && \
    # Ensure Playwright permissions
    chown -R app:app $PLAYWRIGHT_BROWSERS_PATH && \
    chmod -R 750 $PLAYWRIGHT_BROWSERS_PATH && \
    # Create and set permissions for DBus and temp directories
    chown -R app:app /tmp/dbus /var/run/dbus && \
    chmod -R 755 /tmp/dbus /var/run/dbus

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

# Health check configuration
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:${PORT}/diagnostics/logfire-health || exit 1

# Command to run the application with Gunicorn for better performance
CMD ["/usr/local/bin/start-with-dbus.sh", "gunicorn", "--bind", ":8000", "--workers", "1", "--threads", "8", \
    "--timeout", "0", "--keep-alive", "65", "--worker-class", "uvicorn.workers.UvicornWorker", \
    "src.main:app"]