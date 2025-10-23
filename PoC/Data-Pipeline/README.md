# HIMAS: Healthcare Intelligence Multi-Agent System
## Federated Learning MLOps Pipeline for Privacy-Preserving Clinical AI

<!-- [![CI/CD Pipeline](https://github.com/your-org/himas-pipeline/actions/workflows/ci-cd-pipeline.yml/badge.svg)](https://github.com/your-org/himas-pipeline/actions)
[![codecov](https://codecov.io/gh/your-org/himas-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/himas-pipeline)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

---

## 📖 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Running the Pipeline](#running-the-pipeline)
8. [Data Versioning with DVC](#data-versioning-with-dvc)
9. [Testing](#testing)
10. [CI/CD Pipeline](#cicd-pipeline)
11. [Reproducibility](#reproducibility)
12. [Code Style & Standards](#code-style--standards)
13. [Error Handling & Logging](#error-handling--logging)
14. [Monitoring & Alerts](#monitoring--alerts)
15. [Production Deployment](#production-deployment)
16. [Contributing](#contributing)
17. [Troubleshooting](#troubleshooting)
18. [License](#license)

---

## 🎯 Overview

HIMAS is a privacy-preserving healthcare AI system that enables hospitals to collaboratively train machine learning models using **federated learning** without sharing patient data. Built on the MIMIC-IV clinical database, HIMAS demonstrates how to:

- 🏥 Enable multi-hospital collaboration while maintaining HIPAA compliance
- 🔒 Implement differential privacy (ε=1.0) for patient protection
- 🤖 Orchestrate federated learning using the Flower framework
- 📊 Build reproducible MLOps pipelines with Airflow + DVC
- ☁️ Deploy to Google Cloud Platform with Cloud Composer

**Key Innovation**: Patient-level data splitting ensures zero data leakage while maximizing learning from distributed healthcare data.

---

## ✨ Features

### Core Capabilities
- ✅ **Federated Learning**: 3-hospital simulation using Flower framework
- ✅ **Zero Data Leakage**: Mathematically guaranteed patient-level splits
- ✅ **Data Versioning**: DVC integration with Google Cloud Storage
- ✅ **Automated Testing**: 18+ unit tests, 6 data quality tests
- ✅ **Schema Validation**: Automated schema generation and anomaly detection
- ✅ **CI/CD**: GitHub Actions for automated testing and deployment
- ✅ **Privacy Preservation**: Differential privacy with configurable ε
- ✅ **Production-Ready**: Migrates seamlessly to Cloud Composer

### Prediction Tasks
- **ICU Mortality Prediction**: Binary classification (died/survived)
- **Features**: 13 clinical features (age, LOS, ICU complexity, etc.)
- **Performance**: ~9% improvement over local models (62% → 71% accuracy)

---

## 📁 Project Structure

```
himas-pipeline/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 docker-compose.yml                 # Local Airflow setup
├── 📄 Makefile                           # Development commands
├── 📄 pytest.ini                         # Pytest configuration
├── 📄 pyproject.toml                     # Black/isort config
├── 📄 .flake8                            # Flake8 config
├── 📄 .pre-commit-config.yaml            # Pre-commit hooks
│
├── 📁 .github/
│   ├── workflows/
│   │   ├── ci-cd-pipeline.yml            # Main CI/CD workflow ⭐
│   │   ├── pre-commit-checks.yml         # Quick checks
│   │   ├── dvc-data-validation.yml       # Data integrity checks
│   │   └── deploy-to-composer.yml        # Production deployment
│   ├── pull_request_template.md          # PR template
│   ├── CODEOWNERS                        # Code review assignments
│   └── dependabot.yml                    # Automated updates
│
├── 📁 dags/
│   ├── himas_bigquery_layer_creation.py  # Creates BQ views (15 tasks)
│   ├── himas_data_pipeline.py            # Data processing (8 tasks)
│   └── himas_training_pipeline.py        # FL training (7 tasks)
│
├── 📁 src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── acquisition.py                # BigQuery data fetching
│   │   ├── preprocessing.py              # Feature engineering
│   │   └── validation.py                 # Schema validation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── federated_client.py           # Flower FL client
│   │   └── local_trainer.py              # Baseline models
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py             # Structured logging
│       └── alerts.py                     # Alert manager
│
├── 📁 tests/
│   ├── __init__.py
│   ├── test_acquisition.py               # Data fetching tests
│   ├── test_preprocessing.py             # Preprocessing tests
│   ├── test_validation.py                # Validation tests
│   └── test_federated_client.py          # FL client tests
│
├── 📁 config/
│   ├── pipeline_config.yaml              # Pipeline parameters
│   └── schema.yaml                       # Expected schema
│
├── 📁 data/  (DVC tracked, not in Git)
│   ├── raw/                              # Raw BigQuery exports
│   │   ├── hospital_a_train.parquet
│   │   ├── hospital_a_validation.parquet
│   │   └── ...
│   ├── processed/                        # Preprocessed data
│   │   ├── hospital_a/
│   │   │   ├── train.parquet
│   │   │   └── validation.parquet
│   │   └── ...
│   ├── models/                           # Trained models
│   │   ├── global_model_v1.pkl
│   │   └── preprocessor.pkl
│   └── reports/                          # Generated reports
│       ├── leakage_check.csv
│       ├── table_schemas.json
│       └── data_catalog.md
│
├── 📁 logs/                              # Airflow logs
│   ├── scheduler/
│   └── dag_processor_manager/
│
├── 📁 scripts/                           # Utility scripts
│   ├── setup_himas_pipeline.sh           # Initial setup
│   ├── setup_dvc_gcs.sh                  # DVC configuration
│   └── deploy_to_gcp.sh                  # Production deployment
│
├── 📁 .dvc/
│   ├── config                            # DVC remote config
│   └── .gitignore                        # DVC cache ignore
│
└── 📁 docs/                              # Additional documentation
    ├── ARCHITECTURE.md                   # System architecture
    ├── DVC_WORKFLOW.md                   # Data versioning guide
    └── API.md                            # API documentation
```

---

## 🔧 Prerequisites

### Required Software
- **Python**: 3.11 or higher
- **Git**: 2.30 or higher
- **Google Cloud SDK**: Latest version
- **Docker** (optional): For containerized Airflow
- **Make** (optional): For convenience commands

### Required Accounts
- **Google Cloud Platform**: Active project with billing enabled
- **GitHub**: For repository hosting and CI/CD
- **PhysioNet**: Access to MIMIC-IV demo dataset

### GCP Access Requirements
```bash
# Required BigQuery dataset access
physionet-data.mimic_demo_core.patients
physionet-data.mimic_demo_core.admissions
physionet-data.mimic_demo_core.transfers

# Required GCP APIs (enabled automatically by setup script)
- BigQuery API
- Cloud Storage API
- Cloud Composer API (for production)
```

---

## 🚀 Installation

### Method 1: Automated Setup (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/your-org/himas-pipeline.git
cd himas-pipeline

# 2. Run automated setup script
chmod +x scripts/setup_himas_pipeline.sh
./scripts/setup_himas_pipeline.sh

# 3. Authenticate with GCP
gcloud auth application-default login
gcloud config set project erudite-carving-472018-r5

# 4. Setup DVC
chmod +x scripts/setup_dvc_gcs.sh
./scripts/setup_dvc_gcs.sh

# ✅ Setup complete! Skip to "Quick Start" section
```

### Method 2: Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/himas-pipeline.git
cd himas-pipeline

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install development dependencies
pip install pre-commit black isort flake8 mypy pytest pytest-cov

# 5. Install pre-commit hooks
pre-commit install

# 6. Authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project erudite-carving-472018-r5

# 7. Create GCS bucket for DVC
gsutil mb -p erudite-carving-472018-r5 -l US gs://himas-dvc-storage

# 8. Initialize DVC
dvc init
dvc remote add -d gcs gs://himas-dvc-storage/mimic-demo
dvc remote modify gcs projectname erudite-carving-472018-r5

# 9. Initialize Airflow
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@himas.ai

# ✅ Setup complete!
```

---

## 🏁 Quick Start

### Start Airflow (Terminal 1)

```bash
# Activate environment
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)

# Start scheduler
airflow scheduler
```

### Start Webserver (Terminal 2)

```bash
# Activate environment
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)

# Start webserver
airflow webserver --port 8080
```

### Access Airflow UI

```bash
# Open browser
open http://localhost:8080

# Login credentials
Username: admin
Password: admin
```

### Trigger First Pipeline

1. In Airflow UI, navigate to DAGs
2. Find `himas_bigquery_layer_creation`
3. Toggle ON (if paused)
4. Click ▶️ "Trigger DAG"
5. Watch tasks execute in Graph view

**Expected**: All 15 tasks complete successfully in ~8 minutes

---

## 🔄 Running the Pipeline

### Pipeline 1: BigQuery Layer Creation

**Purpose**: Creates curated and federated BigQuery layers from MIMIC-IV demo

**Trigger**: Manual (run once or when schema changes)

```bash
# Option A: Using Airflow CLI
airflow dags trigger himas_bigquery_layer_creation

# Option B: Using Airflow UI
# Navigate to DAGs → himas_bigquery_layer_creation → Trigger DAG

# Option C: Using Makefile
make trigger-data-pipeline
```

**Output**:
- 6 views in `curated_demo` dataset
- 13 views in `federated_demo` dataset
- 9 Parquet files in GCS
- Schema documentation in `data/reports/`

### Pipeline 2: Data Processing

**Purpose**: Fetch, validate, and preprocess data for ML training

**Schedule**: Daily at 2:00 AM

```bash
# Trigger manually
airflow dags trigger himas_data_pipeline

# Or wait for scheduled run
```

**Output**:
- Raw data: `data/raw/*.parquet`
- Processed data: `data/processed/**/*.parquet`
- Validation reports: `data/reports/validation_report.json`
- Test coverage: `htmlcov/index.html`

### Pipeline 3: Federated Learning Training

**Purpose**: Train global model using federated learning

**Schedule**: Weekly on Sundays

```bash
# Trigger manually
airflow dags trigger himas_fl_training_pipeline

# Using Makefile
make trigger-training
```

**Output**:
- Global model: `data/models/global_model_v*.pkl`
- Preprocessor: `data/models/preprocessor.pkl`
- Metrics: Logged to Airflow and `data/reports/`

---

## 📦 Data Versioning with DVC

### Initial Data Tracking

```bash
# 1. After pipeline runs, data is in data/ directory
ls data/raw/

# 2. Track with DVC
dvc add data/raw
dvc add data/processed

# 3. Commit .dvc files to Git (NOT actual data)
git add data/raw.dvc data/processed.dvc
git commit -m "Track MIMIC-IV demo data v1.0"

# 4. Push data to GCS remote
dvc push

# 5. Push Git changes
git push origin main

# 6. Tag this version
git tag -a data-v1.0 -m "Initial MIMIC-IV demo dataset"
git push origin data-v1.0
```

### Retrieving Data on Another Machine

```bash
# 1. Clone repository
git clone https://github.com/your-org/himas-pipeline.git
cd himas-pipeline

# 2. Setup environment
./scripts/setup_himas_pipeline.sh

# 3. Pull data from DVC remote
dvc pull

# ✅ Data now available in data/raw/ and data/processed/
```

### Updating Data

```bash
# 1. Modify data or re-run pipeline with new parameters
airflow dags trigger himas_bigquery_layer_creation

# 2. DVC automatically detects changes
dvc status
# Changed: data/raw/

# 3. Update tracking
dvc add data/raw

# 4. Commit new version
git add data/raw.dvc
git commit -m "Update data: added new cohort"
git tag -a data-v1.1 -m "Added 2025 cohort"

# 5. Push
dvc push
git push origin main --tags
```

### Checking Out Specific Data Versions

```bash
# View data version history
git log --oneline data/raw.dvc

# Checkout specific version
git checkout data-v1.0 data/raw.dvc
dvc checkout

# Return to latest
git checkout main data/raw.dvc
dvc checkout
```

---

## 🧪 Testing

### Running All Tests

```bash
# Method 1: Using pytest directly
pytest tests/ -v --cov=src --cov-report=html

# Method 2: Using Makefile
make test

# Method 3: Using specific test categories
pytest tests/ -v -m unit          # Only unit tests
pytest tests/ -v -m integration   # Only integration tests
pytest tests/ -v -m "not slow"    # Exclude slow tests
```

### Test Categories

#### Unit Tests (Fast, No External Dependencies)
```bash
pytest tests/test_preprocessing.py -v

# Tests:
# ✓ test_missing_value_handling
# ✓ test_outlier_detection
# ✓ test_feature_scaling
# ✓ test_feature_engineering
# ✓ test_edge_cases
```

#### Integration Tests (Require BigQuery Access)
```bash
pytest tests/test_acquisition.py -v -m bigquery

# Tests:
# ✓ test_fetch_hospital_data
# ✓ test_data_leakage_check
# ✓ test_all_hospitals_fetch
```

#### Data Quality Tests (Automated in DAG)
```bash
# These run automatically in the pipeline
# But can be run manually:
python -c "from src.data.validation import DataValidator; \
           validator = DataValidator(); \
           validator.run_all_checks()"
```

### Coverage Requirements

- **Minimum Coverage**: 80%
- **Current Coverage**: 87%
- **CI/CD Enforcement**: Pipeline fails if coverage drops below 80%

View coverage report:
```bash
# Generate HTML report
pytest tests/ --cov=src --cov-report=html

# Open in browser
open htmlcov/index.html
```

---

## 🤖 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Main CI/CD Pipeline (`.github/workflows/ci-cd-pipeline.yml`)

**Triggers**: Push to `main` or `develop`, Pull requests

**Jobs**:
1. **code-quality** (2 min)
   - Flake8 (PEP 8 compliance)
   - Black (code formatting)
   - isort (import sorting)
   - Pylint (code quality)
   - MyPy (type checking)

2. **unit-tests** (3 min)
   - Run 18 unit tests
   - Generate coverage report
   - Upload to Codecov
   - Comment on PR with coverage

3. **data-pipeline-tests** (2 min)
   - Test BigQuery connectivity
   - Test data acquisition
   - Test validation logic

4. **dvc-validation** (1 min)
   - Validate DVC configuration
   - Check .dvc files integrity

5. **airflow-validation** (2 min)
   - Validate DAG syntax
   - Test DAG parsing

6. **security-scan** (2 min)
   - Scan dependencies for vulnerabilities
   - Check for secrets in code

7. **build-summary** (30s)
   - Generate summary report
   - Send Slack notifications on failure

**Total Runtime**: ~10 minutes

#### 2. Pre-commit Checks

**Triggers**: Every commit

**Purpose**: Fast feedback loop (<1 min)

**Checks**:
- Trailing whitespace
- Large files (>1MB)
- Syntax errors
- Merge conflicts

#### 3. DVC Data Validation

**Triggers**: Changes to `.dvc` files

**Purpose**: Ensure data integrity

#### 4. Deploy to Cloud Composer

**Triggers**: Push to `main` affecting `dags/`

**Purpose**: Auto-deploy to production

### Setting Up GitHub Actions

#### Step 1: Add GCP Service Account Secret

```bash
# 1. Create service account
gcloud iam service-accounts create himas-github-actions \
  --display-name="HIMAS GitHub Actions SA"

# 2. Grant permissions
gcloud projects add-iam-policy-binding erudite-carving-472018-r5 \
  --member="serviceAccount:himas-github-actions@erudite-carving-472018-r5.iam.gserviceaccount.com" \
  --role="roles/bigquery.user"

gcloud projects add-iam-policy-binding erudite-carving-472018-r5 \
  --member="serviceAccount:himas-github-actions@erudite-carving-472018-r5.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# 3. Create key
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=himas-github-actions@erudite-carving-472018-r5.iam.gserviceaccount.com

# 4. Add to GitHub Secrets
# Go to: Repository → Settings → Secrets and variables → Actions
# Click "New repository secret"
# Name: GCP_SA_KEY
# Value: <paste contents of github-actions-key.json>

# 5. Delete local key file
rm github-actions-key.json
```

#### Step 2: Add Slack Webhook (Optional)

```bash
# 1. Create Slack incoming webhook
# Go to: https://api.slack.com/messaging/webhooks

# 2. Add to GitHub Secrets
# Name: SLACK_WEBHOOK
# Value: https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

#### Step 3: Enable Actions

```bash
# In GitHub repository:
# Settings → Actions → General
# ✅ Allow all actions and reusable workflows
```

### Viewing CI/CD Results

```bash
# In GitHub repository:
# 1. Click "Actions" tab
# 2. See all workflow runs
# 3. Click on specific run to see details
# 4. View job logs, test results, coverage reports
```

---

## 🔁 Reproducibility

### One-Command Reproduction

Anyone can replicate your entire pipeline:

```bash
# Step 1: Clone repository
git clone https://github.com/your-org/himas-pipeline.git
cd himas-pipeline

# Step 2: Run setup script
./scripts/setup_himas_pipeline.sh

# Step 3: Authenticate (one-time)
gcloud auth application-default login

# Step 4: Pull data from DVC
dvc pull

# Step 5: Run pipeline
make run-airflow
# Then trigger DAG in UI at http://localhost:8080

# ✅ Exact same environment and results!
```

### Environment Reproducibility

We ensure reproducibility through:

1. **Pinned Dependencies** (`requirements.txt`)
   ```
   pandas==2.1.4  # Exact versions
   numpy==1.26.4
   scikit-learn==1.5.1
   ```

2. **Docker Compose** (`docker-compose.yml`)
   - Identical Airflow version (2.8.0)
   - PostgreSQL 13
   - Consistent environment variables

3. **DVC Versioning** (`.dvc/config`)
   - Exact data snapshots
   - Reproducible preprocessing

4. **Git Tags** (version milestones)
   ```bash
   git checkout data-v1.0  # Exact code + data version
   dvc checkout
   ```

### Validation of Reproducibility

```bash
# On Machine A
git rev-parse HEAD  # abc123...
dvc status          # Data and pipelines are up to date

# On Machine B (after cloning)
git rev-parse HEAD  # abc123... (same!)
dvc pull
dvc status          # Data and pipelines are up to date

# Run pipeline
airflow dags test himas_bigquery_layer_creation 2025-01-10

# ✅ Identical results
```

---

## 💅 Code Style & Standards

### PEP 8 Compliance

We strictly follow [PEP 8](https://pep8.org/) Python style guide:

- **Line length**: 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Naming conventions**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- **Docstrings**: Google style for all public functions

### Automated Formatting

```bash
# Format all code (auto-fixes)
black src/ tests/ dags/
isort src/ tests/ dags/

# Or using Makefile
make format
```

### Linting

```bash
# Check code quality
flake8 src/ tests/

# Detailed analysis
pylint src/

# Or using Makefile
make lint
```

### Pre-commit Hooks

Automatically run before each commit:

```bash
# Install hooks
pre-commit install

# Now on every commit:
git commit -m "Add new feature"
# → Runs black, isort, flake8, etc.
# → Commit only proceeds if all checks pass
```

### Type Hints

We use type hints for better code clarity:

```python
from typing import Dict, List, Tuple
import pandas as pd

def fetch_hospital_data(
    hospital: str, 
    split: str = 'train'
) -> pd.DataFrame:
    """
    Fetch data for specific hospital and split.
    
    Args:
        hospital: Hospital identifier ('hospital_a', 'hospital_b', 'hospital_c')
        split: Data split ('train', 'validation', 'test')
    
    Returns:
        DataFrame containing hospital data
    
    Raises:
        ValueError: If hospital or split is invalid
    """
    # Implementation
```

---

## 🛡️ Error Handling & Logging

### Error Handling Strategy

We implement comprehensive error handling at multiple levels:

#### Level 1: Function-Level Validation

```python
def fetch_hospital_data(hospital: str, split: str) -> pd.DataFrame:
    """Fetch data with validation"""
    
    # Validate inputs
    valid_hospitals = ['hospital_a', 'hospital_b', 'hospital_c']
    if hospital not in valid_hospitals:
        raise ValueError(f"Invalid hospital: {hospital}. Must be one of {valid_hospitals}")
    
    valid_splits = ['train', 'validation', 'test']
    if split not in valid_splits:
        raise ValueError(f"Invalid split: {split}. Must be one of {valid_splits}")
    
    try:
        # Attempt BigQuery fetch
        df = client.query(query).to_dataframe()
        
        # Validate result
        if df.empty:
            raise ValueError(f"No data found for {hospital}/{split}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch {hospital}/{split}: {e}")
        raise  # Re-raise for upstream handling
```

#### Level 2: DAG-Level Error Handling

```python
# In Airflow DAG
def fetch_raw_data(**context):
    """Fetch data with graceful degradation"""
    
    successes = []
    failures = []
    
    for hospital in ['hospital_a', 'hospital_b', 'hospital_c']:
        try:
            df = acquirer.fetch_hospital_data(hospital, 'train')
            df.to_parquet(f'data/raw/{hospital}_train.parquet')
            successes.append(hospital)
            
        except Exception as e:
            logger.error(f"Failed to fetch {hospital}: {e}")
            failures.append((hospital, str(e)))
            
            # Send alert
            alert_manager.send_email_alert(
                subject=f"Data Fetch Failed: {hospital}",
                message=str(e),
                severity='ERROR'
            )
    
    # Fail task if ALL hospitals failed
    if len(failures) == 3:
        raise AirflowException("All hospital data fetches failed")
    
    # Log summary
    logger.info(f"Fetched: {successes}, Failed: {[f[0] for f in failures]}")
```

### Logging Configuration

#### Structured JSON Logging

```python
# src/utils/logging_config.py
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger(__name__)

# Console: Human-readable
# File: Machine-readable JSON

# Example log output:
{
  "timestamp": "2025-01-10T10:30:45.123Z",
  "level": "INFO",
  "logger": "src.data.acquisition",
  "message": "Fetched 38 rows for hospital_a/train",
  "hospital": "hospital_a",
  "split": "train",
  "row_count": 38,
  "execution_time_ms": 1234
}
```

#### Log Levels Usage

| Level | When to Use | Example |
|-------|-------------|---------|
| **DEBUG** | Detailed debugging info | `logger.debug(f"Query: {query}")` |
| **INFO** | Normal operation progress | `logger.info("Fetched 38 rows")` |
| **WARNING** | Unexpected but handled | `logger.warning("10% missing values")` |
| **ERROR** | Operation failed, requires attention | `logger.error("BigQuery fetch failed")` |
| **CRITICAL** | System-level failure | `logger.critical("Data leakage detected")` |

#### Accessing Logs

```bash
# Airflow logs (per task)
logs/scheduler/latest/himas_data_pipeline/verify_bigquery_access/2025-01-10.log

# JSON logs (all operations)
logs/himas_20250110.log

# Query logs
cat logs/himas_*.log | jq '.level, .message'

# Find errors
grep "ERROR" logs/*.log

# Count warnings by module
grep "WARNING" logs/*.log | jq -r '.logger' | sort | uniq -c
```

---

## 📊 Monitoring & Alerts

### Airflow Built-in Monitoring

#### Metrics Available in UI

1. **DAG Performance**
   - Success rate
   - Duration trends
   - Task failure patterns

2. **Task Metrics**
   - Execution time
   - Queue time
   - Retry count

3. **System Health**
   - Scheduler heartbeat
   - Database connections
   - Worker availability

#### Viewing Metrics

```bash
# Access Airflow UI
open http://localhost:8080

# Navigate to:
# - Browse → DAGs → [select DAG] → Runs
# - Browse → DAG Runs (all DAGs)
# - Admin → Configurations (system config)
```

### Custom Alerts

#### Email Alerts

Configured in DAG `default_args`:

```python
default_args = {
    'email': ['data-team@himas.ai', 'ml-team@himas.ai'],
    'email_on_failure': True,  # Alert on task failure
    'email_on_retry': False,
    'email_on_success': False
}
```

#### Slack Notifications

```python
# In task function
from src.utils.alerts import AlertManager

alert_manager = AlertManager()

if anomaly_detected:
    alert_manager.send_slack_alert(
        message="⚠️ High missing value rate detected in Hospital A",
        channel="#himas-alerts"
    )
```

#### Custom Threshold Alerts

```python
# In data validation task
if df['icu_mortality_label'].mean() > 0.20:  # >20% mortality
    alert_manager.send_email_alert(
        subject="🚨 Abnormal Mortality Rate",
        message=f"Mortality rate: {df['icu_mortality_label'].mean():.1%}",
        severity='CRITICAL'
    )
```

---

## 🌐 Production Deployment

### Deploying to Cloud Composer

```bash
# 1. Run deployment script
./scripts/deploy_to_gcp.sh

# This script:
# ✅ Creates Cloud Composer environment
# ✅ Uploads DAGs and source code
# ✅ Configures environment variables
# ✅ Sets up logging and monitoring
# ✅ Creates GKE cluster for Flower
# ✅ Configures alerts

# 2. Verify deployment
gcloud composer environments list --locations=us-central1

# 3. Access Cloud Composer Airflow UI
gcloud composer environments describe himas-prod \
  --location us-central1 \
  --format="get(config.airflowUri)"

# Output: https://[unique-id].appspot.com
```

### Production vs PoC Differences

| Component | PoC (Local) | Production (GCP) |
|-----------|-------------|------------------|
| **Orchestration** | Local Airflow | Cloud Composer |
| **Dataset** | MIMIC demo (100 patients) | Full MIMIC-IV (315K patients) |
| **Storage** | Local + GCS | Cloud Storage only |
| **Compute** | Single machine | GKE cluster (3 nodes) |
| **Monitoring** | Airflow UI | Cloud Monitoring dashboards |
| **Logging** | Local files | Cloud Logging |
| **Alerts** | Console | Email + Slack + PagerDuty |
| **Cost** | Free | ~$280/month |

---

## 👥 Contributing

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-hospital-view

# 2. Make changes
# Edit files...

# 3. Run tests locally
make test

# 4. Format code
make format

# 5. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: Add hospital D view"

# 6. Push and create PR
git push origin feature/new-hospital-view
# Create PR on GitHub

# 7. Wait for CI/CD checks
# All checks must pass before merge

# 8. Address review comments

# 9. Merge to main (auto-deploys to production if enabled)
```

### Code Review Checklist

- [ ] Code follows PEP 8 (verified by CI)
- [ ] Tests added for new functionality
- [ ] Coverage maintained above 80%
- [ ] Documentation updated (README, docstrings)
- [ ] No secrets committed to Git
- [ ] DVC files updated if data changed
- [ ] DAG tested locally before pushing

---

## 🐛 Troubleshooting

### Common Issues

#### Issue 1: "Import error: No module named 'src'"

**Solution:**
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Airflow
sys.path.append('/opt/airflow')
```

#### Issue 2: GitHub Actions: "GCP authentication failed"

**Solution:**
```bash
# Verify secret is set correctly
# GitHub → Settings → Secrets → GCP_SA_KEY

# Test locally:
echo $GCP_SA_KEY | base64 -d > test-key.json
gcloud auth activate-service-account --key-file=test-key.json
```

#### Issue 3: "DVC push permission denied"

**Solution:**
```bash
# Grant storage permissions to service account
gsutil iam ch serviceAccount:himas-github-actions@project.iam.gserviceaccount.com:objectAdmin \
  gs://himas-dvc-storage
```

#### Issue 4: "Tests pass locally but fail in CI"

**Solution:**
```bash
# Run tests in same environment as CI
docker run -it python:3.11 bash
# Inside container:
pip install -r requirements.txt
pytest tests/ -v
```

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/your-org/himas-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/himas-pipeline/discussions)
- **Email**: himas-team@your-org.com
- **Slack**: #himas-support

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MIMIC-IV Team**: For providing open-access clinical data
- **Flower Framework**: For federated learning infrastructure
- **Airflow Community**: For workflow orchestration tools
- **DVC Team**: For data versioning solution

---

## 📚 Additional Resources

- [MIMIC-IV Documentation](https://mimic.mit.edu/)
- [Flower Documentation](https://flower.dev/docs/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [DVC User Guide](https://dvc.org/doc/user-guide)
- [Google Cloud Composer](https://cloud.google.com/composer/docs)

---