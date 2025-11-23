# -----------------------------
# 1. Base Python image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# 2. Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Set environment variables
# -----------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -----------------------------
# 4. Copy requirements first for better caching
# -----------------------------
COPY api/requirements.txt .

# -----------------------------
# 5. Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 6. Copy the entire project
# -----------------------------
COPY . /app
COPY models/ ./models/
COPY metrics/ ./metrics

# -----------------------------
# 7. Configure Streamlit for headless mode
# -----------------------------
RUN mkdir -p /root/.streamlit
RUN echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nport = 8501" > /root/.streamlit/config.toml

# -----------------------------
# 8. Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# 9. Run Streamlit app
# -----------------------------
CMD ["streamlit", "run", "api/app.py", "--server.port=8501", "--server.address=0.0.0.0"]