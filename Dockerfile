# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-dev --no-root && \
    poetry build

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /app/.venv ./.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application
COPY . .

# Set up logging and data directories
RUN mkdir -p /app/logs /app/data/raw /app/data/processed /app/data/embeddings

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]