FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy application code
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY scripts ./scripts

# Copy dependency files
COPY pyproject.toml README.md requirements.txt ./

# Install Python dependencies
# RUN pip install --upgrade pip setuptools wheel && \
#     pip install -e . && \
RUN pip install -r requirements.txt

# Create artifact directories
RUN mkdir -p data artifacts/models artifacts/metrics logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
