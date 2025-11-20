# -----------------------------
# 1. Base Python image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# 2. Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------
# 4. Disable DVC analytics
# -----------------------------
ENV DVC_NO_ANALYTICS=1

# -----------------------------
# 5. Copy the entire project (api/, predictor.py, etc.)
# -----------------------------
COPY . /app

# -----------------------------
# 6. Install Python dependencies
# -----------------------------
# Install dependencies
COPY api/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install dvc[azure]


# -----------------------------
# 7. Pull model files from Azure Blob via DVC
# -----------------------------
# Azure credentials will be passed at runtime via --env-file
RUN dvc pull || echo "⚠️ DVC pull failed — will retry at runtime"

# -----------------------------
# 8. Expose Flask port
# -----------------------------
EXPOSE 5000

# -----------------------------
# 9. Run the Flask server
# -----------------------------
CMD ["python", "api/app.py"]
