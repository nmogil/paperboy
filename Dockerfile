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

# Copy requirements files and install Python dependencies
COPY requirements.txt requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

# Final stage
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install only the necessary Playwright dependencies
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
    rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install Chromium browser as root (skip other browsers)
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/ms-playwright
RUN mkdir -p $PLAYWRIGHT_BROWSERS_PATH && \
    playwright install chromium --with-deps

# Create config directory and ensure it exists
RUN mkdir -p /app/config

# Create a non-root user and group called 'app'
RUN groupadd --system app && useradd --system --gid app app

# Create and set permissions for home directory and cache
RUN mkdir -p /home/app && \
    chown -R app:app /home/app && \
    chown -R app:app $PLAYWRIGHT_BROWSERS_PATH

# Set appropriate permissions for application and config
RUN chown -R app:app /app && \
    chmod -R 755 /app/config

# Copy the application code
COPY . .

# Set HOME environment variable
ENV HOME=/home/app

# Switch to the non-root user
USER app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 