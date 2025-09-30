FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY docker_client.py .

# Expose port for Flower server
EXPOSE 8080

# Default: run server (can be overridden in docker-compose)
CMD ["python", "-c", "from src.federated.flower_server import start_federated_server; start_federated_server()"]