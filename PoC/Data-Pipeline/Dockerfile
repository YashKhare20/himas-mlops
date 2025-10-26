FROM apache/airflow:3.1.0

# Set working directory
WORKDIR /opt/airflow

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir apache-airflow==${AIRFLOW_VERSION} -r requirements.txt

# Copy all project files into the image
COPY dags/ /opt/airflow/dags/
COPY config/ /opt/airflow/config/

# Create necessary runtime directories
RUN mkdir -p /opt/airflow/logs \
    && mkdir -p /opt/airflow/plugins

# Set permissions (important for Airflow)
# Make the entrypoint script executable
USER root
RUN chown -R airflow:root /opt/airflow && \
    chmod -R 775 /opt/airflow

RUN apt-get update && apt-get install -y git

# Switch to airflow user
USER airflow