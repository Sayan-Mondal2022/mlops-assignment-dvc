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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 4. Copy requirements first for better caching
# -----------------------------
COPY api/requirements.txt /app/requirements.txt

# -----------------------------
# 5. Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------
# 6. Copy the entire project
# -----------------------------
COPY . /app

# -----------------------------
# 7. Create necessary directories
# -----------------------------
RUN mkdir -p models/cpu

# -----------------------------
# 8. Expose Flask port
# -----------------------------
EXPOSE 5001

# -----------------------------
# 9. Run the Flask server
# -----------------------------
CMD ["python", "app.py"]