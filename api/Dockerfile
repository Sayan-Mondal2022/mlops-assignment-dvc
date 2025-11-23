# -----------------------------
# 1. Base Python image
# -----------------------------
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# -----------------------------
# Install dependencies
# -----------------------------
COPY api/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application files
# -----------------------------
COPY . /app
COPY models /app/models
COPY metrics /app/metrics

# -----------------------------
# Streamlit config
# -----------------------------
RUN mkdir -p /root/.streamlit
RUN echo "[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nport = 8501" > /root/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "api/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
