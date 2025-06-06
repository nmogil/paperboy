# Lightweight Dockerfile for Cloud Run deployment
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.lightweight.txt ./
RUN pip install --no-cache-dir -r requirements.lightweight.txt

# Create a non-root user with specific UID/GID
RUN groupadd --system --gid 10001 app && \
    useradd --system --uid 10001 --gid app --no-create-home app && \
    mkdir -p /app/config /app/data && \
    # Set strict permissions
    chown -R app:app /app && \
    chmod -R 750 /app

# Copy application files with strict permissions
COPY --chown=app:app src/ /app/src/
RUN chmod -R 550 /app/src  # Read and execute only

# Mount points for config and data
RUN mkdir -p /app/config /app/data && \
    chown app:app /app/config /app/data && \
    chmod 750 /app/config /app/data

# Set HOME and switch to non-root user
ENV HOME=/tmp
USER 10001:10001

# Expose the port the app runs on
EXPOSE 8080

# Health check for Cloud Run
HEALTHCHECK --interval=43200s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run the application
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT} --log-level info"]