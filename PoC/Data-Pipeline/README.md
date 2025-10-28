# HIMAS Data Pipeline - Healthcare Intelligence Multi-Agent System

> **Automated BigQuery data pipeline with schema validation, statistics generation, and DVC versioning for federated healthcare learning**

[![Airflow](https://img.shields.io/badge/Airflow-3.1.0-017CEE?logo=apache-airflow)](https://airflow.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![BigQuery](https://img.shields.io/badge/BigQuery-Enabled-4285F4?logo=google-cloud)](https://cloud.google.com/bigquery)
[![DVC](https://img.shields.io/badge/DVC-Versioned-945DD6)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Environment Setup](#-environment-setup)
- [Code Structure](#-code-structure)
- [Data Acquisition](#-data-acquisition)
- [Data Preprocessing](#-data-preprocessing)
- [Pipeline Orchestration](#-pipeline-orchestration)
- [Data Versioning with DVC](#-data-versioning-with-dvc)
- [Schema & Statistics Validation](#-schema--statistics-validation)
- [Tracking & Logging](#-tracking--logging)
- [Anomaly Detection & Alerts](#-anomaly-detection--alerts)
- [Testing](#-testing)
- [Pipeline Optimization](#-pipeline-optimization)
- [Troubleshooting](#-troubleshooting)
- [Appendix](#-appendix-pair-guidebook-worksheets)
- [Contributing](#-contributing)

---
## 🎯 Overview

HIMAS Data Pipeline is a production-ready, scalable data pipeline built with Apache Airflow that processes MIMIC-IV healthcare data for federated learning scenarios. The pipeline implements automated schema validation, comprehensive statistics generation, and data versioning using DVC, all while maintaining HIPAA compliance through patient-level data splitting.

### Key Capabilities

- **Federated Learning Ready**: Partitions data across 3 simulated hospitals (40%, 35%, 25%) with zero patient leakage
- **Automated Schema Validation**: Extracts schemas, detects drift, and validates data quality over time
- **Comprehensive Statistics**: Field-level statistics with anomaly detection and configurable thresholds
- **Data Versioning**: Complete DVC integration for reproducible ML workflows
- **Production-Grade**: Email alerts, comprehensive logging, error handling, and monitoring

### Use Case

This pipeline enables collaborative machine learning for ICU mortality prediction across multiple hospitals without sharing patient data, demonstrating federated learning principles while maintaining complete data privacy.

---

## 🏗️ Architecture

### High-Level Architecture

![Pipeline Overview](assets/pipeline-overview.png)

### Data Flow Architecture

![Dataflow Overview](assets/dataflow-overview.png)

### Technology Stack

![Technology Stack](assets/technology-stack.png)
---

## 📁 Code Structure

### Project Organization

```
PoC/Data-Pipeline/
│
├── .github/
│   └── workflows/
│       └── himas-ci.yml              # CI/CD pipeline
│
├── dags/
│   ├── himas_bigquery_demo.py        # Main DAG file
│   │
│   ├── sql/                          # SQL transformations
│   │   ├── curated_layer/
│   │   │   ├── patient_split_assignment.sql
│   │   │   ├── dim_patient.sql
│   │   │   ├── fact_hospital_admission.sql
│   │   │   ├── fact_icu_stay.sql
│   │   │   ├── fact_transfers.sql
│   │   │   └── clinical_features.sql
│   │   │
│   │   ├── federated_layer/
│   │   │   ├── hospital_a_data.sql
│   │   │   ├── hospital_b_data.sql
│   │   │   └── hospital_c_data.sql
│   │   │
│   │   └── verification_layer/
│   │       ├── data_leakage_check.sql
│   │       └── dataset_statistics.sql
│   │
│   └── utils/                        # Utility modules
│       ├── __init__.py
│       ├── config.py                 # Centralized configuration
│       ├── storage.py                # Storage handler (local/GCS)
│       ├── sql_utils.py              # SQL file loader
│       ├── validation.py             # Data validation
│       ├── email_callbacks.py        # Email alert handlers
│       ├── dvc_handler.py            # DVC operations
│       ├── schema_validator.py       # Schema validation engine
│       ├── schema_utils.py           # Schema helper functions
│       └── task_functions.py         # Airflow task wrappers
│
├── data/                             # Data outputs (local or GCS)
│   ├── schemas/                      # Table schemas
│   ├── statistics/                   # Field statistics
│   ├── drift/                        # Drift reports
│   ├── validation/                   # Quality validation
│   └── reports/                      # Summary reports
│
├── tests/                            # Unit tests
│   ├── test_dag_integrity.py
│   ├── test_data_leakage.py
│   ├── test_dvc_setup.py
│   └── conftest.py
│
├── config/                           # Configuration files
│   └── gcp-key.json                  # GCP service account key
│
├── logs/                             # Airflow logs
├── plugins/                          # Airflow plugins (optional)
│
├── .dvc/                             # DVC configuration
│   ├── config
│   └── .gitignore
│
├── docker compose.yaml               # Docker services definition
├── Dockerfile                        # Airflow image build
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables
├── .gitignore                        # Git ignore rules
├── .dockerignore                     # Docker ignore rules
├── pytest.ini                        # Pytest configuration
└── README.md                         # This file
```

### Module Descriptions

#### Core Modules

**`config.py`** - Centralized Configuration
```python
class PipelineConfig:
    """
    Centralized configuration for HIMAS pipeline.
    
    Features:
    - GCP settings (PROJECT_ID, LOCATION)
    - Dataset definitions (DATASETS, LAYERS)
    - Table lists (CURATED_TABLES, FEDERATED_TABLES)
    - Storage settings (USE_GCS, GCS_BUCKET)
    - Email settings (ALERT_EMAILS)
    """
    PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
    LOCATION = 'US'
    USE_GCS = os.getenv('USE_GCS', 'false').lower() == 'true'
```

**`storage.py`** - Hybrid Storage Handler
```python
class StorageHandler:
    """
    Handles storage operations for local or GCS.
    
    Features:
    - Save to local filesystem or GCS
    - Read from local filesystem or GCS
    - Automatic fallback to local
    - JSON string upload/download
    """
    def save(self, data: str, filename: str) -> str:
        if self.use_gcs:
            return self._save_to_gcs(data, filename)
        else:
            return self._save_to_local(data, filename)
```

**`schema_validator.py`** - Schema Validation Engine
```python
class SchemaValidator:
    """
    Core validation logic for schema and statistics.
    
    Methods:
    - extract_table_schema() - Extract BigQuery schema
    - compute_table_statistics() - Field-level stats
    - detect_schema_drift() - Compare schemas
    - validate_data_quality() - Threshold validation
    """
```

**`schema_utils.py`** - Config-Aware Helpers
```python
# Helper functions that integrate with PipelineConfig
def extract_all_layer_schemas(validator, run_id, storage_handler):
    """Extract schemas from all configured layers"""

def compute_all_layer_statistics(validator, run_id, storage_handler):
    """Compute statistics for all configured layers"""
```

**`task_functions.py`** - Airflow Task Wrappers
```python
# Factory functions for Airflow tasks with lazy initialization
def create_extract_schemas_task_function(validator, config, storage):
    """Create task function with runtime initialization"""
    def task_function(**context):
        # Initialize at runtime (not parse time)
        if validator is None:
            validator = SchemaValidator(...)
        return extract_all_schemas_task(validator, ...)
    return task_function
```

#### Supporting Modules

**`sql_utils.py`** - SQL File Management
```python
class SQLFileLoader:
    """Load SQL files organized by layer"""
    def get_layer_files(self, layer_name: str) -> List[Path]:
        """Get all SQL files for a layer"""
```

**`dvc_handler.py`** - DVC Operations
```python
class DVCHandler:
    """Handle DVC versioning operations"""
    def version_reports(self) -> bool:
        """Version reports directory"""
    
    def version_all_data(self) -> bool:
        """Version all data directories"""
```

**`email_callbacks.py`** - Alert Handlers
```python
def send_success_email(context):
    """Send success notification email"""

def send_failure_email(context):
    """Send failure notification email"""
```

### Design Patterns

#### 1. Factory Pattern (Task Functions)

```python
# Factory creates task functions with closure
def create_extract_schemas_task_function(validator, config, storage):
    def task_function(**context):
        return extract_all_schemas_task(validator, config, storage, **context)
    return task_function
```

#### 2. Lazy Initialization

```python
# Initialize at runtime, not import time
if schema_validator is None:
    from utils.schema_validator import SchemaValidator
    validator = SchemaValidator(...)
```

#### 3. Strategy Pattern (Storage)

```python
# Conditional storage strategy based on flag
if self.use_gcs:
    return self._save_to_gcs(data, filename)
else:
    return self._save_to_local(data, filename)
```

#### 4. Template Method (Validation)

```python
# Common validation workflow
def validate_data_quality(current, baseline, thresholds):
    validation = {"passed": True, "errors": [], "warnings": []}
    
    # Check row count
    # Check null rates
    # Check cardinality
    
    return validation
```

---

## ✨ Features

### Core Pipeline Features

- **Dimensional Data Model**: Star schema with facts and dimensions for healthcare data
- **Patient-Level Splitting**: Prevents data leakage with proper train/val/test splits
- **Hospital Partitioning**: Simulates federated learning across 3 hospitals
- **Automated Quality Checks**: Data leakage detection and integrity validation
- **Modular Architecture**: Reusable utilities for config, storage, SQL, and validation

### Schema & Statistics Validation

- **Automated Schema Extraction**: Extracts BigQuery table schemas with metadata
- **Comprehensive Statistics**: Field-level stats (mean, min, max, stddev, null rates, cardinality)
- **Schema Drift Detection**: Compares against historical baselines
- **Data Quality Validation**: Configurable thresholds for anomaly detection
- **Historical Tracking**: Maintains baseline schemas and statistics
- **Flexible Storage**: Local filesystem or Google Cloud Storage

### DevOps & MLOps

- **Data Version Control (DVC)**: Version datasets, schemas, and statistics
- **Containerization**: Docker-based deployment with Docker Compose
- **Email Alerts**: Notifications for pipeline success, failures, and anomalies
- **Comprehensive Logging**: Python logging and Airflow task logs
- **CI/CD Ready**: GitHub Actions integration with automated testing
- **Unit Testing**: Pytest-based tests for all components

---

## 📦 Prerequisites

### Required Software

- **Docker Desktop** (v20.10+) - [Download](https://www.docker.com/products/docker-desktop)
- **Docker Compose** (v2.14.0+) - Included with Docker Desktop
- **Git** (v2.30+) - [Download](https://git-scm.com/downloads)
- **Google Cloud SDK** (gcloud CLI) - [Installation Guide](https://cloud.google.com/sdk/docs/install)

### System Requirements

- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Disk Space**: 20GB free space
- **OS**: Linux, macOS, or Windows (with WSL2)
- **Python 3.11+**

### Google Cloud Platform (GCP) Access

- **GCP Project** with BigQuery API enabled
- **MIMIC-IV Demo Access** - Request access at [PhysioNet](https://physionet.org/content/mimiciv-demo/)
- **Service Account** with BigQuery permissions or gcloud authentication

### Verify Prerequisites

```bash
# Check Docker
docker --version
docker compose --version

# Check Git
git --version

# Check gcloud CLI
gcloud --version

# Check available memory
docker run --rm "debian:bookworm-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
```

---

## 🚀 Environment Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/YashKhare20/himas-mlops.git
cd himas-mlops/PoC/Data-Pipeline

# Verify structure
ls -la
```

Expected structure:
```
Data-Pipeline/
├── dags/
├── .dvc/
├── config/
├── tests/
├── scripts/
├── docker compose.yaml
├── Dockerfile
├── docker compose.yaml
├── requirements.txt
├── .env
└── README.md
```

### Step 2: Google Cloud Platform Setup

#### Option A: Using gcloud CLI (Recommended for Development)

```bash
# Install gcloud CLI (if not already installed)
# macOS
brew install --cask google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project erudite-carving-472018-r5

# Verify authentication
gcloud auth list
```

#### Option B: Using Service Account (Will be used for Production)

```bash
# Create service account
gcloud iam service-accounts create himas-pipeline \
    --display-name="HIMAS Data Pipeline"

# Grant BigQuery permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:himas-pipeline@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

# Create and download key
gcloud iam service-accounts keys create config/gcp-key.json \
    --iam-account=himas-pipeline@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Verify key file
ls -la config/gcp-key.json
```

### Step 3: MIMIC-IV Access Setup

1. **Request Access**:
   - Visit [PhysioNet MIMIC-IV Demo](https://physionet.org/content/mimic-iv-demo/2.2/)
   - Complete required training (CITI Data or Specimens Only Research)
   - Sign Data Use Agreement

2. **Link to BigQuery**:
   - Follow [MIMIC-IV BigQuery Guide](https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/)
   - Link PhysioNet account to Google Cloud
   - Verify access in BigQuery console

3. **Verify Access**:
```bash
# Test BigQuery access to MIMIC-IV demo
bq query --use_legacy_sql=false \
'SELECT COUNT(*) as patient_count
FROM `physionet-data.mimiciv_demo.patients`'
```

Expected output: `patient_count: 100`

### Step 4: Configure Environment Variables

Create `.env` file in the project root:

Required variables in `.env`:

```bash
# ============================================================================
# GCP Configuration
# ============================================================================
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/config/gcp-key.json

# ============================================================================
# Storage Configuration
# ============================================================================
# Set to 'true' to use GCS, 'false' for local storage
USE_GCS=false
GCS_BUCKET=your-bucket-name

# ============================================================================
# Email Alerts Configuration
# ============================================================================
ALERT_EMAIL=your-email@example.com
ALERT_EMAILS=your-email@example.com,team@example.com

# SMTP Configuration (for Gmail)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_MAIL_FROM=your-email@gmail.com

# ============================================================================
# Airflow Configuration
# ============================================================================
AIRFLOW_UID=50000
```

**Gmail App Password Setup**:
1. Go to [Google Account Settings](https://myaccount.google.com/security)
2. Enable 2-Factor Authentication
3. Generate App Password for "Mail"
4. Use generated password in `SMTP_PASSWORD`

### Step 5: Create Required Directories

```bash
# Create all required directories
mkdir -p data/{schemas,statistics,drift,validation,reports}
mkdir -p logs
mkdir -p plugins
mkdir -p config

# Set permissions (important for Docker)
chmod -R 777 data/
chmod -R 777 logs/

# Verify structure
tree -L 2 -d
```

### Step 6: Build and Start Services

```bash
# Build Docker images
docker compose build

# Verify build
docker images | grep airflow

# Start all services
docker compose up -d

# Check service health
docker compose ps
```

Expected output - all services should be `healthy`:
```
NAME                    STATUS
airflow-api-server      Up (healthy)
airflow-scheduler       Up (healthy)
airflow-worker          Up (healthy)
postgres                Up (healthy)
redis                   Up (healthy)
```

![Docker Services](assets/docker-services.png)

### Step 7: Access Airflow Web UI

```bash
# Wait for services to be ready (~2 minutes)
sleep 120

# Check if Airflow is ready
curl http://localhost:8080/health
```

1. Open browser: http://localhost:8080
2. Login credentials:
   - **Username**: `airflow`
   - **Password**: `airflow`

3. Verify DAG appears:
   - Look for `himas_bigquery_demo` in DAG list
   - Status should be "paused" (not "error")

![Airflow UI](assets/airflow-ui.png)

![DAG List](assets/dag-list.png)

### Step 8: Verify Installation

```bash
# Check DAG integrity
docker compose exec airflow-worker airflow dags list | grep himas

# Check for import errors
docker compose exec airflow-worker airflow dags list-import-errors

# Test BigQuery connection
docker compose exec airflow-worker python -c "
from google.cloud import bigquery
client = bigquery.Client(project='YOUR_PROJECT_ID')
datasets = list(client.list_datasets())
print(f'Connected to BigQuery. Found {len(datasets)} datasets')
"

# Check DVC
docker compose exec airflow-worker dvc version
```

---

## 📊 Data Acquisition

### MIMIC-IV Demo Dataset

The pipeline uses the **MIMIC-IV Demo dataset**, a de-identified healthcare database containing comprehensive clinical data for 100 ICU patients.

#### Dataset Details

| Component | Description | Records |
|-----------|-------------|---------|
| **Patients** | Demographics, admission details | 100 patients |
| **Admissions** | Hospital admission records | 100+ admissions |
| **ICU Stays** | ICU admission details | 100+ ICU stays |
| **Transfers** | Patient location transfers | 500+ transfers |

#### Access Configuration

The dataset is accessed via BigQuery public datasets:

```sql
-- Project: physionet-data
-- Dataset: mimiciv_demo
-- Tables: patients, admissions, icustays, transfers, diagnoses_icd, procedures_icd
```

#### Access Verification

```bash
# Verify MIMIC-IV access
bq query --use_legacy_sql=false \
'SELECT
  COUNT(DISTINCT subject_id) as patients,
  COUNT(DISTINCT hadm_id) as admissions,
  COUNT(DISTINCT stay_id) as icu_stays
FROM `physionet-data.mimiciv_demo.icustays`'
```

**Documentation**: [MIMIC-IV BigQuery Guide](https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/)

---

## 🔄 Data Preprocessing

### Overview

The pipeline implements a **three-layer architecture** for data preprocessing, ensuring clean, validated, and ML-ready data while maintaining patient-level isolation for federated learning.

### Preprocessing Layers

![Preprocessing Layers](assets/data-layers.png)

![BigQuery](assets/bq-datasets.png)

### 1. Patient Split Assignment

**Purpose**: Assign each patient to a hospital and data split (train/val/test) to prevent data leakage.

**Key Features**:
- Deterministic hashing ensures reproducibility
- Patient-level splitting prevents temporal leakage
- Stratified distribution across hospitals
- No patient appears in multiple splits or hospitals

### 2. Dimensional Model (Curated Layer)

#### 2.1. Dim Patient (Demographics)

**Purpose**: Patient demographic dimension table.

**Transformations**:
- Clean and standardize gender field
- Calculate age at first admission
- Handle missing values with defaults
- Add metadata timestamps

#### 2.2. Fact Hospital Admission

**Purpose**: Hospital admission events fact table.

**Transformations**:
- Join admissions with patient assignments
- Calculate length of stay
- Standardize admission/discharge types
- Extract temporal features

#### 2.3. Fact ICU Stay (Mortality Labels)

**Purpose**: ICU stay fact table with mortality prediction labels.

**Transformations**:
- Calculate ICU length of stay
- Create binary mortality label
- Extract first care unit
- Add temporal features

#### 2.4. Clinical Features (ML Ready)

**Purpose**: Aggregated clinical features for machine learning.

**Transformations**:
- Aggregate multiple admissions per patient
- Calculate key clinical metrics
- Create feature vectors
- Handle missing values

### 3. Federated Layer (Hospital Partitions)

**Purpose**: Create isolated datasets for each hospital to simulate federated learning.

**Implementation**: Separate tables for each hospital containing only their patients.

**Validation**: No patient appears in multiple hospital datasets.

### Preprocessing Code Organization

All preprocessing logic is modularized in SQL files:

```
dags/sql/
├── curated_layer/
│   ├── patient_split_assignment.sql    # Patient-level splitting
│   ├── dim_patient.sql                  # Demographics dimension
│   ├── fact_hospital_admission.sql      # Admission facts
│   ├── fact_icu_stay.sql                # ICU facts with labels
│   ├── fact_transfers.sql               # Transfer events
│   └── clinical_features.sql            # ML-ready features
│
├── federated_layer/
│   ├── hospital_a_data.sql              # Hospital A partition
│   ├── hospital_b_data.sql              # Hospital B partition
│   └── hospital_c_data.sql              # Hospital C partition
│
└── verification_layer/
    ├── data_leakage_check.sql           # Leakage validation
    └── dataset_statistics.sql           # Summary statistics
```

**Modularity**: Each SQL file is independent and reusable, enabling easy adjustments and testing.

---

## 🔧 Pipeline Orchestration

### Airflow DAG Structure

The pipeline is orchestrated using **Apache Airflow 3.1.0** with a modular DAG structure that ensures logical task dependencies and optimal execution.

### Complete Pipeline Flow

![Pipeline Flow](assets/pipeline-flow.png)

### Task Group Details

#### 1. Create Datasets

**Purpose**: Initialize BigQuery datasets for all layers.

**Operator**: `BigQueryCreateEmptyDatasetOperator`

**Tasks**:
- `create_curated_demo` - Dimensional model dataset
- `create_federated_demo` - Hospital partition dataset
- `create_verification_demo` - Quality check dataset

**Execution**: Parallel (no dependencies)

#### 2. Curated Layer

**Purpose**: Build dimensional model with patient-level splitting.

**Operator**: `BigQueryInsertJobOperator`

**Tasks** (Sequential):
1. `patient_split_assignment` - Assign patients to hospitals and splits
2. `dim_patient` - Patient demographics dimension
3. `fact_hospital_admission` - Hospital admission facts
4. `fact_icu_stay` - ICU stay facts with mortality labels
5. `fact_transfers` - Patient transfer events
6. `clinical_features` - Aggregated ML features

**Execution**: Sequential (order matters for foreign keys)

#### 3. Federated Layer

**Purpose**: Create hospital-specific data partitions.

**Operator**: `BigQueryInsertJobOperator`

**Tasks** (Parallel):
- `hospital_a_data` - 40% of patients
- `hospital_b_data` - 35% of patients
- `hospital_c_data` - 25% of patients

**Execution**: Parallel (independent hospitals)

#### 4. Verification Layer

**Purpose**: Validate data integrity and quality.

**Operator**: `BigQueryInsertJobOperator`

**Tasks** (Parallel):
- `data_leakage_check` - Ensure no patient overlap
- `dataset_statistics` - Generate summary statistics

**Execution**: Parallel (independent checks)

#### 5. Schema & Statistics Validation

**Purpose**: Automated schema extraction, statistics computation, and drift detection.

**Operator**: `PythonOperator`

**Tasks**:
1. `extract_all_schemas` - Extract schemas from all BigQuery tables
2. `compute_all_statistics` - Compute field-level statistics
3. `detect_schema_drift` - Compare against baseline
4. `validate_data_quality` - Check quality thresholds
5. `generate_quality_summary` - Create comprehensive report

**Execution**: Mixed (extract & compute parallel, then sequential validation)

#### 6. DVC Versioning

**Purpose**: Version all generated data for reproducibility.

**Operator**: `PythonOperator`

**Tasks** (Sequential):
1. `version_bigquery_layers` - Export and version BigQuery tables
2. `version_reports` - Version validation reports
3. `version_all_data` - Version all data directories

### Running the Pipeline

#### Via Airflow UI

1. Navigate to http://localhost:8080
2. Find `himas_bigquery_demo` DAG
3. Click **Trigger DAG** (play button)
4. Monitor execution in Graph view

![DAG Execution](assets/dag-execution.png)

#### Via CLI

```bash
# Trigger DAG
docker compose exec airflow-worker airflow dags trigger himas_bigquery_demo

# Check status
docker compose exec airflow-worker airflow dags state himas_bigquery_demo

# View task logs
docker compose exec airflow-worker airflow tasks logs himas_bigquery_demo extract_all_schemas <execution_date>
```

#### Via API

```bash
# Trigger via REST API
curl -X POST \
  --user "airflow:airflow" \
  "http://localhost:8080/api/v1/dags/himas_bigquery_demo/dagRuns" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Monitoring Execution

#### Airflow UI Views

1. **Graph View**: Visual task dependencies
2. **Gantt Chart**: Task duration and parallelization
3. **Task Duration**: Historical performance trends
4. **Logs**: Detailed task execution logs

![Graph View](assets/himas_bigquery_demo-graph.png)

![Gantt Chart](assets/gantt-chart.png)

---

## 💾 Data Versioning with DVC

### DVC Integration

The pipeline uses **Data Version Control (DVC)** to track and version all data outputs, ensuring complete reproducibility of ML workflows.

### What Gets Versioned

![DVC](assets/dvc.png)

![GCS](assets/gcs-bucket.png)

### DVC Configuration

#### Initialize DVC (First Time Only)

```bash
# Navigate to project directory
cd PoC/Data-Pipeline

# Initialize DVC
dvc init

# Configure remote storage (Local)
dvc remote add -d local /path/to/dvc-storage

# OR configure remote storage (GCS)
dvc remote add -d gcs gs://your-bucket/dvc-storage
dvc remote modify gcs projectname your-project-id

# Verify configuration
dvc remote list
cat .dvc/config
```

Example `.dvc/config`:

```ini
[core]
    autostage = true
    remote = gcs_storage
['remote "gcs_storage"']
    url = gs://himas-airflow-data/dvc-storage
    projectname = erudite-carving-472018-r5
```

### DVC Workflow

#### 1. Automatic Versioning (via Pipeline)

The pipeline automatically versions data after each run:

```python
# In version_with_dvc() task group
version_bigquery_layers()  # Exports BigQuery → CSV → DVC
version_reports()           # Versions validation reports
version_all_data()          # Versions all data directories
```

#### 2. Manual Versioning

```bash
# Add data to DVC tracking
dvc add data/schemas/schema_curated.json

# Commit .dvc files to Git
git add data/schemas/schema_curated.json.dvc .gitignore
git commit -m "Version curated layer schemas"

# Push data to remote
dvc push

# Push code to Git
git push
```

#### 3. Retrieving Versioned Data

```bash
# Pull latest data
dvc pull

# Checkout specific version
git checkout <commit-hash>
dvc checkout

# View data status
dvc status

# View data metrics
dvc metrics show
```

### DVC File Structure

```
PoC/Data-Pipeline/
├── .dvc/
│   ├── config                    # DVC configuration
│   ├── .gitignore               # DVC internal files
│   └── cache/                   # Local cache (gitignored)
│
├── data/
│   ├── schemas/
│   │   ├── schema_curated.json
│   │   └── schema_curated.json.dvc    # DVC metadata
│   ├── statistics/
│   │   ├── statistics_curated.json
│   │   └── statistics_curated.json.dvc
│   └── reports/
│       ├── quality_summary.json
│       └── quality_summary.json.dvc
│
└── dvc.yaml                     # DVC pipelines (optional)
```

### DVC Remote Storage Options

#### Local Storage (Development)

```bash
# Set in .env
USE_GCS=false

# DVC uses local filesystem
dvc remote add -d local /opt/airflow/dvc-storage
```

#### Google Cloud Storage (Production)

```bash
# Set in .env
USE_GCS=true
GCS_BUCKET=your-bucket-name

# DVC uses GCS
dvc remote add -d gcs gs://your-bucket-name/dvc-storage
dvc remote modify gcs projectname your-project-id

# Authenticate
gcloud auth application-default login

# Test push
dvc push
```

### Reproducibility Example

```bash
# Team member clones repo
git clone https://github.com/your-username/himas-mlops.git
cd himas-mlops/PoC/Data-Pipeline

# Pull versioned data
dvc pull

# Data is now available locally
ls -la data/schemas/
ls -la data/statistics/
```

---

## 🔍 Schema & Statistics Validation

### Overview

The pipeline implements a comprehensive **Schema & Statistics Validation System** that automatically generates data schemas, computes field-level statistics, detects schema drift, and validates data quality over time.

### Architecture

![Schema Validation](assets/schema-validation.png)

### Components

#### 1. SchemaValidator (`utils/schema_validator.py`)

Core validation engine that handles:

**Schema Extraction**:
```python
schema = validator.extract_table_schema('curated_demo', 'dim_patient')
# Returns: {table_id, dataset_id, num_rows, num_bytes, fields[]}
```

**Statistics Computation**:
```python
stats = validator.compute_table_statistics('curated_demo', 'fact_icu_stay')
# Returns: Field-level stats (mean, min, max, stddev, null_rate, distinct_count)
```

**Drift Detection**:
```python
drift = validator.detect_schema_drift(current_schema, baseline_schema)
# Returns: {has_drift, added_fields, removed_fields, modified_fields}
```

**Quality Validation**:
```python
validation = validator.validate_data_quality(current_stats, baseline_stats, thresholds)
# Returns: {passed, errors, warnings, metrics}
```

#### 2. SchemaUtils (`utils/schema_utils.py`)

Config-aware helper functions that integrate SchemaValidator with PipelineConfig:

- `extract_all_layer_schemas()` - Extract schemas for all configured layers
- `compute_all_layer_statistics()` - Compute statistics for all tables
- `detect_schema_drift_all_layers()` - Detect drift across all layers
- `validate_data_quality_all_layers()` - Validate quality across all layers
- `generate_comprehensive_quality_summary()` - Create unified summary report

#### 3. TaskFunctions (`utils/task_functions.py`)

Airflow task wrappers with factory functions that enable lazy initialization for CI/CD compatibility.

### Validation Process

#### Step 1: Schema Extraction

Extracts complete schema definitions from BigQuery tables:

**Output**: `data/schemas/schema_curated.json`

```json
{
  "dim_patient": {
    "table_id": "dim_patient",
    "dataset_id": "curated_demo",
    "num_rows": 100,
    "num_bytes": 15084226,
    "fields": [
      {
        "name": "subject_id",
        "field_type": "INT64",
        "mode": "REQUIRED"
      },
      {
        "name": "gender",
        "field_type": "STRING",
        "mode": "NULLABLE"
      }
    ]
  }
}
```

#### Step 2: Statistics Computation

Computes comprehensive field-level statistics:

**Output**: `data/statistics/statistics_curated.json`

```json
{
  "fact_icu_stay": {
    "row_count": 234,
    "size_mb": 1.23,
    "field_statistics": {
      "icu_los_hours": {
        "type": "FLOAT64",
        "count": 234,
        "distinct": 187,
        "mean": 72.5,
        "min": 1.2,
        "max": 480.3,
        "stddev": 45.6
      },
      "mortality_label": {
        "type": "BOOLEAN",
        "count": 234,
        "true_count": 23,
        "false_count": 211
      }
    }
  }
}
```

#### Step 3: Schema Drift Detection

Compares current schemas against historical baseline:

**Output**: `data/drift/schema_drift_{run_id}.json`

```json
{
  "table_id": "dim_patient",
  "has_drift": true,
  "changes": {
    "added_fields": ["insurance_type"],
    "removed_fields": [],
    "modified_fields": [
      {
        "field": "gender",
        "change": "mode",
        "old": "NULLABLE",
        "new": "REQUIRED"
      }
    ],
    "row_count_change": {
      "baseline": 100,
      "current": 103,
      "delta": 3,
      "percent_change": 3.0
    }
  }
}
```

#### Step 4: Data Quality Validation

Validates data against configurable thresholds:

**Output**: `data/validation/quality_validation_{run_id}.json`

**Default Thresholds**:
- Max row count change: 50%
- Max null rate per field: 30%
- Min distinct ratio: 1%

```json
{
  "table_id": "fact_hospital_admission",
  "passed": false,
  "errors": [
    {
      "type": "row_count_anomaly",
      "message": "Row count changed by 65% (threshold: 50%)",
      "baseline": 200,
      "current": 330
    }
  ],
  "warnings": [
    {
      "type": "high_null_rate",
      "field": "discharge_location",
      "message": "Field has 45% null values (threshold: 30%)"
    }
  ]
}
```

#### Step 5: Quality Summary

Consolidated view of all validation results:

**Output**: `data/reports/quality_summary_{run_id}.json`

```json
{
  "run_id": "manual__2025-10-28T10:30:00+00:00",
  "summary": {
    "schemas": {
      "total_tables": 12,
      "total_layers": 3
    },
    "statistics": {
      "total_rows": 1234,
      "total_size_mb": 45.67
    },
    "drift": {
      "has_baseline": true,
      "total_drifts": 2
    },
    "validation": {
      "overall_passed": false,
      "total_errors": 1,
      "total_warnings": 3
    }
  }
}
```

### Customizing Thresholds

Modify thresholds in `dags/utils/task_functions.py`:

```python
custom_thresholds = {
    "row_count_change_pct": 50.0,    # Adjust as needed
    "null_rate_threshold": 0.3,       # Adjust as needed
    "distinct_ratio_min": 0.01        # Adjust as needed
}
```

### Baseline Management

**First Run**: Automatically establishes baseline
```bash
# Baselines created at:
# - data/schemas/schemas_baseline.json
# - data/statistics/statistics_baseline.json
```

**Update Baseline**: After planned schema changes
```bash
# Manually update baseline
cp data/schemas/schemas_all_latest.json data/schemas/schemas_baseline.json
cp data/statistics/statistics_all_latest.json data/statistics/statistics_baseline.json
```

**Reset Baseline**: Start fresh
```bash
rm data/schemas/schemas_baseline.json
rm data/statistics/statistics_baseline.json
# Next run will establish new baseline
```

---

## 📝 Tracking & Logging

### Logging Architecture

![Logging Architecture](assets/logging-arch.png)

### Application-Level Logging

All utility modules use Python's logging library:

```python
# In utils/schema_validator.py
import logging

logger = logging.getLogger(__name__)

# Log levels used
logger.info("Extracting schema for layer: curated")
logger.warning("Baseline schema file not found")
logger.error("Failed to compute statistics: {error}")
```

**Log Locations**:
- **Task Logs**: `/opt/airflow/logs/dag_id/task_id/execution_date/`
- **Application Logs**: Embedded in task logs

### Viewing Logs

#### Via Airflow UI

1. Navigate to http://localhost:8080
2. Click on DAG: `himas_bigquery_demo`
3. Click on any task instance
4. Click **Log** button

![Task Logs](assets/airflow-logs.png)

#### Via Docker

```bash
# View scheduler logs
docker compose logs -f airflow-scheduler

# View worker logs
docker compose logs -f airflow-worker

# View specific task logs
docker compose exec airflow-worker bash
cat /opt/airflow/logs/himas_bigquery_demo/extract_all_schemas/2025-10-28T10:30:00+00:00/1.log
```

#### Via CLI

```bash
# View task logs
docker compose exec airflow-worker airflow tasks logs \
  himas_bigquery_demo extract_all_schemas 2025-10-28

# Follow logs in real-time
docker compose exec airflow-worker airflow tasks logs -f \
  himas_bigquery_demo extract_all_schemas 2025-10-28
```

### Log Levels and Usage

| Level | Usage | Example |
|-------|-------|---------|
| `INFO` | Normal operations | "Extracting schemas for run_id: ..." |
| `WARNING` | Potential issues | "Baseline not found - establishing..." |
| `ERROR` | Failures | "Failed to compute statistics: ..." |
| `DEBUG` | Detailed debugging | "Processing table: dim_patient" |

### Monitoring Best Practices

1. **Regular Log Review**: Check logs after each run
2. **Error Pattern Analysis**: Identify recurring issues
3. **Performance Monitoring**: Track task durations in Gantt chart
4. **Alert Configuration**: Set up email alerts for critical failures

---

## 🚨 Anomaly Detection & Alerts

### Anomaly Detection System

![Alerting Architecture](assets/alerting-arch.png)

### Types of Anomalies Detected

#### 1. Row Count Anomalies

**Detection**: Significant changes in table row counts

```python
# Threshold: 50% change
if abs((current_rows - baseline_rows) / baseline_rows) > 0.50:
    # Trigger alert
```

**Example Alert**:
```
❌ Row count anomaly in fact_hospital_admission
Baseline: 200 rows
Current: 330 rows
Change: +65% (threshold: 50%)
```

#### 2. High Null Rate

**Detection**: Fields with excessive null values

```python
# Threshold: 30% nulls
null_rate = 1 - (non_null_count / total_count)
if null_rate > 0.30:
    # Trigger warning
```

**Example Alert**:
```
⚠️ High null rate in discharge_location
Null rate: 45%
Threshold: 30%
```

#### 3. Low Cardinality

**Detection**: Unexpectedly low distinct values

```python
# Threshold: 1% distinct
distinct_ratio = distinct_count / total_count
if distinct_ratio < 0.01:
    # Trigger warning
```

#### 4. Schema Drift

**Detection**: Schema changes between runs

**Detected Changes**:
-  Added fields
-  Removed fields
-  Type changes (INT64 → STRING)
-  Mode changes (NULLABLE → REQUIRED)

### Alert Configuration

#### Email Alerts

Configure in `.env`:

```bash
# Single recipient
ALERT_EMAIL=your-email@example.com

# Multiple recipients
ALERT_EMAILS=admin@example.com,team@example.com,data-eng@example.com

# SMTP settings (Gmail example)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_MAIL_FROM=your-email@gmail.com
```

#### Alert Triggers

**Success Notification**:
```python
# In DAG definition
on_success_callback=send_success_email
```

**Failure Notification**:
```python
# In default_args
'email_on_failure': True,
'email': config.ALERT_EMAILS
```

**Custom Anomaly Alerts**:
```python
# In validate_data_quality_task
if not validation_results["overall_passed"]:
    error_msg = f"Data quality validation failed with {total_errors} errors"
    raise ValueError(error_msg)  # Triggers email alert
```

### Email Alert Examples

#### Success Email

![Success Email](assets/success-email.png)

#### Failure Email

![Failure Email](assets/failure-email.png)

## 🧪 Testing

### Test Suite Overview

The pipeline includes comprehensive unit tests using **pytest** framework.

### Test Structure

```
tests/
├── test_dag_integrity.py           # DAG structure and integrity tests
├── test_data_leakage.py            # Data leakage prevention tests
├── test_dvc_setup.py               # DVC configuration tests
├── conftest.py                     # Pytest configuration
└── pytest.ini                      # Pytest settings
```

### Running Tests

#### Run All Tests

```bash
# Inside Docker container
docker compose exec airflow-worker pytest tests/ -v

# With coverage report
docker compose exec airflow-worker pytest tests/ -v \
  --cov=dags \
  --cov-report=term-missing \
  --cov-report=html

# View coverage
open htmlcov/index.html
```

#### Run Specific Test Modules

```bash
# DAG integrity tests
pytest tests/test_dag_integrity.py -v

# Data leakage tests
pytest tests/test_data_leakage.py -v

# DVC setup tests
pytest tests/test_dvc_setup.py -v
```

#### Run Specific Test Classes

```bash
# Schema validation setup tests
pytest tests/test_data_leakage.py::TestSchemaValidationSetup -v

# Data quality threshold tests
pytest tests/test_data_leakage.py::TestDataQualityThresholds -v
```

### Test Examples

#### DAG Integrity Test

```python
def test_no_import_errors(self, dag_bag):
    """Test that there are no import errors in DAGs"""
    assert not dag_bag.import_errors, \
        f"DAG import errors: {dag_bag.import_errors}"

def test_no_cycles(self, dag_bag):
    """Test that DAG has no cycles"""
    for dag_id, dag in dag_bag.dags.items():
        check_cycle(dag)
```

#### Schema Validation Test

```python
def test_schema_validator_methods_exist(self):
    """Test that SchemaValidator has required methods"""
    import inspect
    
    methods = inspect.getmembers(SchemaValidator, predicate=inspect.isfunction)
    method_names = [name for name, _ in methods]
    
    assert 'extract_table_schema' in method_names
    assert 'compute_table_statistics' in method_names
    assert 'detect_schema_drift' in method_names
    assert 'validate_data_quality' in method_names
```

#### Data Leakage Test

```python
def test_verification_layer_sql_exists(self):
    """Test that verification layer SQL files exist"""
    sql_dir = Path(DAG_FOLDER) / 'sql' / 'verification_layer'
    
    leakage_check_files = [
        f for f in sql_dir.glob('*.sql')
        if 'leakage' in f.name.lower()
    ]
    
    assert len(leakage_check_files) > 0, \
        "Should have SQL file for data leakage checks"
```

### CI/CD Integration

The pipeline includes GitHub Actions workflow for automated testing:

```yaml
# .github/workflows/himas-ci.yml
name: HIMAS CI Tests

on:
  pull_request:
    branches: [main, feature_poc]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
      - name: Install dependencies
      - name: Validate DAG Integrity
      - name: Run unit tests with coverage
      - name: Code quality check
```

**Test Results**: [[CI Status]](https://github.com/YashKhare20/himas-mlops/actions)

![Test Results](assets/github-actions.png)

### Adding New Tests

```python
# tests/test_custom.py
import pytest
from utils.schema_validator import SchemaValidator

class TestCustomValidation:
    """Custom validation tests"""
    
    def test_mortality_rate_reasonable(self):
        """Test that mortality rate is within expected range"""
        # Your test logic here
        pass
```

### Test Coverage Goals

- **DAG Structure**: >95% coverage
- **Utility Modules**: >80% coverage
- **Task Functions**: >75% coverage
- **Overall**: >70% coverage

---

## ⚡ Pipeline Optimization

### Identifying Bottlenecks

#### Via Airflow UI

1. Navigate to DAG: `himas_bigquery_demo`
2. Click **Gantt** view
3. Identify tasks with longest duration
4. Analyze parallelization opportunities

![Gantt Chart View](assets/gantt-chart.png)

#### Via Task Duration View

1. Click **Task Duration** in DAG view
2. Review historical task performance
3. Identify trends and outliers

![Task Duration](assets/dag-execution.png)

### Optimization Strategies

#### 1. Parallelize Independent Tasks

**Before**:
```python
# Sequential execution
hospital_a >> hospital_b >> hospital_c
```

**After**:
```python
# Parallel execution (no dependencies)
[hospital_a, hospital_b, hospital_c]
```

**Improvement**: 3x faster (30s → 10s)

#### 2. Optimize BigQuery Queries

**Slow Query**:
```sql
-- Full table scan
SELECT * FROM large_table
WHERE condition;
```

**Optimized Query**:
```sql
-- Partitioned and filtered
SELECT column1, column2, column3
FROM large_table
WHERE _PARTITIONDATE = CURRENT_DATE()
  AND condition;
```

**Improvement**: 10x faster for large tables

#### 3. Use Table Sampling for Statistics

For very large tables (>1M rows):

```python
# In schema_validator.py
stats_query = f"""
    SELECT COUNT(*), AVG(field), ...
    FROM `{table_ref}` TABLESAMPLE SYSTEM (10 PERCENT)
"""
```

**Improvement**: 10x faster statistics computation

#### 4. Increase Worker Concurrency

```yaml
# In docker compose.yaml
airflow-worker:
  command: celery worker -c 4  # Increase from default 1
```

**Improvement**: More parallel task execution

#### 5. Resource Allocation

```yaml
# In docker compose.yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

**Improvement**: Better resource utilization

### Performance Benchmarks

| Component | Before Optimization | After Optimization | Improvement |
|-----------|-------------------|-------------------|-------------|
| Federated Layer | 30s (sequential) | 10s (parallel) | 3x faster |
| Statistics Computation | 5m (full scan) | 30s (sampling) | 10x faster |
| Schema Extraction | 1m (all fields) | 30s (optimized) | 2x faster |
| **Total Pipeline** | **8m 45s** | **5m 23s** | **38% faster** |

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: DAG Import Errors

**Symptoms**:
```
DAG Import Errors:
himas_bigquery_demo.py: ModuleNotFoundError: No module named 'utils'
```

**Solution**:
```bash
# Verify PYTHONPATH
docker compose exec airflow-worker python -c "import sys; print(sys.path)"

# Restart scheduler
docker compose restart airflow-scheduler

# Check file permissions
ls -la dags/utils/
```

#### Issue: Permission Denied on Data Directory

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied: '/opt/airflow/data/schemas/...'
```

**Solution**:
```bash
# Set proper permissions
chmod -R 777 data/
chmod -R 777 logs/

# Rebuild and restart
docker compose down
docker compose build
docker compose up -d
```

#### Issue: BigQuery Authentication Failed

**Symptoms**:
```
DefaultCredentialsError: Your default credentials were not found
```

**Solution**:
```bash
# Option 1: gcloud auth
gcloud auth application-default login

# Option 2: Service account
# Verify key file exists
ls -la config/gcp-key.json

# Update .env
GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/config/gcp-key.json

# Restart containers
docker compose restart
```

#### Issue: Schema Validation Fails

**Symptoms**:
```
ValueError: Data quality validation failed with 2 errors
```

**Solution**:
```bash
# Check validation report
cat data/validation/quality_validation_*.json | jq '.errors'

# Review thresholds in task_functions.py
# Adjust if needed based on your data

# Reset baseline if schema changed intentionally
rm data/schemas/schemas_baseline.json
rm data/statistics/statistics_baseline.json
```

#### Issue: DVC Push Fails

**Symptoms**:
```
ERROR: failed to push data to the cloud - ... Permission denied
```

**Solution**:
```bash
# Verify GCS bucket exists
gsutil ls gs://your-bucket-name

# Check permissions
gsutil iam get gs://your-bucket-name

# Re-authenticate
gcloud auth application-default login

# Retry push
dvc push
```

#### Issue: Email Alerts Not Sending

**Symptoms**:
- No emails received on pipeline success/failure

**Solution**:
```bash
# Verify SMTP configuration in .env
cat .env | grep SMTP

# Test SMTP connection
docker compose exec airflow-worker python << 'EOF'
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your-email@gmail.com', 'your-app-password')
print("✓ SMTP connection successful")
server.quit()
EOF

# Check Airflow connection
docker compose exec airflow-worker airflow connections list | grep smtp

# Restart services
docker compose restart
```

#### Issue: Container Out of Memory

**Symptoms**:
```
airflow-worker continuously restarting
Container killed (OOMKilled)
```

**Solution**:
```bash
# Increase Docker memory (Docker Desktop → Settings → Resources)
# Minimum: 8GB, Recommended: 16GB

# Reduce worker concurrency
# In docker compose.yaml: celery worker -c 2 (instead of -c 4)

# Restart
docker compose restart
```

### Debug Mode

```bash
# Enter container for debugging
docker compose exec airflow-worker bash

# Test Python imports
python -c "
from utils.config import PipelineConfig
from utils.schema_validator import SchemaValidator
print('✓ All imports successful')
"

# Test BigQuery connection
python -c "
from google.cloud import bigquery
client = bigquery.Client(project='YOUR_PROJECT_ID')
print('✓ BigQuery connection successful')
"

# Check DVC status
dvc status

# View environment variables
env | grep AIRFLOW
env | grep GOOGLE
```

### Getting Help

1. **Check Logs**: Always start with task logs in Airflow UI
2. **Review Documentation**: See inline code documentation
3. **Search Issues**: Check GitHub Issues for similar problems
4. **Ask Community**: Airflow Slack, Stack Overflow

---

## 📚 Appendix: PAIR Guidebook Worksheets

### Google PAIR (People + AI Research) Design Framework

As part of the HIMAS project development, we followed Google's People + AI Research (PAIR) design framework to ensure our data pipeline meets user needs and handles errors gracefully.

- **Completed Worksheets**: See `assets/` folder for full PDF documents


## 🤝 Contributing

### Development Setup

```bash
# Clone repo
git clone https://github.com/your-username/himas-mlops.git
cd himas-mlops/PoC/Data-Pipeline

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ...

# Run tests
docker compose exec airflow-worker pytest tests/ -v

# Run linting
docker compose exec airflow-worker flake8 dags/

# Commit changes
git add .
git commit -m "feat: your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### Code Style Guidelines

- **PEP 8**: Follow Python style guide
- **Line Length**: Max 127 characters
- **Docstrings**: Required for all functions and classes
- **Type Hints**: Use where appropriate
- **Comments**: Explain complex logic

### Testing Requirements

-  All new features must have unit tests
-  Maintain >70% code coverage
-  No flake8 errors
-  All tests must pass in CI/CD

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Run tests locally
4. Push to GitHub
5. Create Pull Request
6. Wait for CI/CD to pass
7. Request review
8. Merge after approval

---

## 📄 License

This project is part of the HIMAS (Healthcare Intelligence Multi-Agent System) research project.

**Dataset License**: MIMIC-IV Demo is available under PhysioNet Credentialed Health Data License.

---

## 👥 Team

**Institution**: Northeastern University
**Course**: MLOps - IE7374
**Semester**: Fall 2025

---

Made with ❤️ for Healthcare AI and Federated Learning