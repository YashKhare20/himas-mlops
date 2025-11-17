# Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


# System dependencies (compiler etc. for some Python deps)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential gcc git \
#   && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git jq bc curl \
  && rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN curl https://sdk.cloud.google.com | bash
ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

# Create working directory
WORKDIR /app

# Copy the entire project into the image
COPY . .

# Install build backend and project dependencies (incl. mlflow etc. via pyproject)
RUN pip install --upgrade pip \
 && pip install hatchling \
 && pip install "flwr[app]" \ 
 && pip install .

# Default command is a shell; docker-compose overrides this
CMD ["bash"]
# Updated Fri Nov 14 17:36:32 EST 2025
# Rebuild Mon Nov 17 10:48:25 EST 2025
