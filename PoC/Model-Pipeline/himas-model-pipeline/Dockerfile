# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies (compiler etc. for some Python deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git \
  && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install build backend and project dependencies
COPY pyproject.toml ./

RUN pip install --upgrade pip \
 && pip install hatchling \
 && pip install .

# Copy the rest of the source code into the image
COPY . .

# Default command is a shell; docker-compose overrides this
CMD ["bash"]
