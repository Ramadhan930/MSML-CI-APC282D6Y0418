FROM python:3.12.3-slim

WORKDIR /app

# Salin model dan dependencies
COPY model /app/model
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (default MLflow models serve pakai 8080, kita sesuaikan)
EXPOSE 8080

# Jalankan MLflow serve untuk model yang sudah disimpan
CMD ["mlflow", "models", "serve", "-m", "model", "--host", "0.0.0.0", "--port", "8080"]