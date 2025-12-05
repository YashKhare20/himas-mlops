# ---------------------------
# 1. Base Image
# ---------------------------
    FROM python:3.10-slim

    # ---------------------------
    # 2. Install required system packages
    # ---------------------------
    RUN apt-get update && apt-get install -y \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # ---------------------------
    # 3. Create work directory
    # ---------------------------
    WORKDIR /app
    
    # ---------------------------
    # 4. Copy requirements
    # ---------------------------
    COPY app/requirements.txt /app/requirements.txt


    
    # ---------------------------
    # 5. Install Python dependencies
    # ---------------------------
    RUN pip install --no-cache-dir -r /app/requirements.txt
    
    # ---------------------------
    # 6. Copy application code
    # ---------------------------
    COPY . /app
    
    # ---------------------------
    # 7. Expose port required by Vertex AI (predict container)
    # ---------------------------
    EXPOSE 8080
    
    # ---------------------------
    # 8. Start FastAPI server (Vertex AI expects port 8080)
    # ---------------------------
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
    