FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages like chromadb/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Pre-install CPU-only PyTorch to avoid massive CUDA binary downloads (saves ~500MB and speeds up build)
RUN pip install --no-cache-dir --retries 10 --default-timeout=2000 torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --retries 10 --default-timeout=1000 -r requirements.txt

# Copy application source code
COPY . .

# Expose API port
EXPOSE 8000

# Start the FastAPI application via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
