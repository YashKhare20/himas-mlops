FROM google/cloud-sdk:slim

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir \
    "tensorflow>=2.11.1,<2.18.0" \
    && pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Set TensorFlow environment variables
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Copy application code
COPY *.py ./
COPY startup.sh ./
RUN chmod +x startup.sh

# Create directory for model artifacts
RUN mkdir -p /app/local_model

# Expose port
ENV PORT=8080
EXPOSE 8080

# Run startup script
CMD ["./startup.sh"]
