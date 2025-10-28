FROM apache/airflow:3.1.0

# Set permissions (important for Airflow)
# Make the entrypoint script executable
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary runtime directories
RUN mkdir -p /opt/airflow/data/schemas \
    /opt/airflow/data/statistics/ \
    /opt/airflow/data/drift/ \
    /opt/airflow/data/validation/ \
    /opt/airflow/data/reports/ \
    /opt/airflow/logs \
    /opt/airflow/plugins

# Copy all project files into the image
COPY dags/ /opt/airflow/dags/
COPY config/ /opt/airflow/config/

RUN chown -R airflow:root /opt/airflow && \
    chmod -R 775 /opt/airflow

# Switch to airflow user
USER airflow

# Set working directory
WORKDIR /opt/airflow

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt