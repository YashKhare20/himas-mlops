# HIMAS Model Deployment

Complete deployment infrastructure for the HIMAS federated ICU mortality prediction model on Google Cloud Platform.

---

## Deployment Architecture

**Platform:** Google Cloud Platform (GCP)  
**Service:** Cloud Run (serverless containers)  
**Automation:** Cloud Build + GitHub integration  
**Monitoring:** Cloud Functions + Cloud Scheduler + BigQuery  

```
GitHub Push → Cloud Build Pipeline → Cloud Run Service
                                           ↓
                              Cloud Function (every 6 hrs)
                                           ↓
                              BigQuery Monitoring Table
```

---

## Directory Structure

```
PoC/Model-Serving/
├── app/                    # Cloud Run application
│   ├── main.py            # FastAPI prediction API
│   ├── predictor.py       # Model inference + preprocessing
│   ├── preprocessing_loader.py  # Load scaler/encoders
│   ├── startup.sh         # Download model from GCS
│   ├── Dockerfile         # Container definition
│   └── requirements.txt   # Python dependencies
├── cloud-functions/       # Monitoring infrastructure
│   └── model-monitor/     # Automated performance testing
│       ├── main.py        # Cloud Function code
│       └── requirements.txt
├── testing/               # Manual testing scripts
│   ├── test_deployed_model.py  # Test with real data
│   └── test_predictions.sh     # Traffic generation
└── README.md             # This file
```

---

## Deployment Features

### 1. Automated Deployment Pipeline

**Trigger:** Push to `main` branch on GitHub

**Pipeline Actions:**
1. Train federated model on BigQuery data (3 hospitals)
2. Evaluate model performance on test set
3. **Quality Gates:**
   - Accuracy must be ≥ 85%
   - Recall must be ≥ 70%
   - Bias metrics within acceptable ranges
   - New model must not be >2% worse than previous
4. If all checks pass: Deploy to Cloud Run automatically
5. If any check fails: Stop pipeline, keep old model deployed

**Result:** Only validated, high-quality models reach production

### 2. Model Serving (Cloud Run)

**Endpoint:** `https://himas-prediction-service-erudite-carving-472018-r5.us-central1.run.app`

**Features:**
- Downloads latest model from GCS at startup
- Applies preprocessing (StandardScaler, LabelEncoders)
- Serves predictions via `/predict` endpoint
- Auto-scales from 1 to 10 instances based on traffic
- 2 CPU cores, 2GB memory per instance

**Preprocessing Steps:**
1. Load feature_order, scaler, label_encoders from GCS
2. Impute missing values (0 for numeric, 'Unknown' for categorical)
3. Scale numeric features (z-score normalization)
4. Encode categorical features (integer encoding)
5. Feed to model in correct feature order

### 3. Automated Monitoring

**Schedule:** Every 6 hours (12 AM, 6 AM, 12 PM, 6 PM EST)

**Process:**
1. Cloud Scheduler triggers Cloud Function
2. Function queries 100 random test samples from BigQuery
3. Makes predictions via Cloud Run API
4. Calculates metrics: accuracy, recall, precision, confusion matrix
5. Stores results in `production_monitoring.model_performance` table
6. Compares against thresholds:
   - Accuracy < 85% → Alert for retraining
   - Recall < 70% → Alert for retraining

**Monitoring Table Schema:**
```sql
production_monitoring.model_performance (
  timestamp TIMESTAMP,
  samples_tested INT64,
  accuracy FLOAT64,
  recall FLOAT64,
  precision FLOAT64,
  tp INT64,  -- True positives
  fp INT64,  -- False positives
  fn INT64,  -- False negatives
  tn INT64   -- True negatives
)
```

---

## Setup Instructions

### Prerequisites Setup

```bash
# Set project
export PROJECT_ID="erudite-carving-472018-r5"
export PROJECT_NUMBER="1089649594993"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com
```

### 1. Create Storage Infrastructure

```bash
# GCS bucket for models
gsutil mb -l us-central1 gs://himas-mlops-models

# BigQuery monitoring dataset
bq mk --dataset --location=us-central1 ${PROJECT_ID}:production_monitoring

# Monitoring table
bq mk --table ${PROJECT_ID}:production_monitoring.model_performance \
  timestamp:TIMESTAMP,samples_tested:INTEGER,accuracy:FLOAT,recall:FLOAT,precision:FLOAT,tp:INTEGER,fp:INTEGER,fn:INTEGER,tn:INTEGER
```

### 2. Deploy Cloud Run Service

**Automatic (via Cloud Build):**
```bash
# Trigger will deploy automatically on GitHub push
git push origin main
```

**Manual:**
```bash
cd PoC/Model-Serving/app

# Build and push container
gcloud builds submit \
  --tag gcr.io/${PROJECT_ID}/himas-cloudrun:latest

# Deploy to Cloud Run
gcloud run deploy himas-prediction-service \
  --image gcr.io/${PROJECT_ID}/himas-cloudrun:latest \
  --region us-central1 \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1
```

### 3. Deploy Monitoring Function

```bash
cd PoC/Model-Serving/cloud-functions/model-monitor

# Deploy function
gcloud functions deploy model-monitor \
  --gen2 \
  --runtime python311 \
  --region us-central1 \
  --source . \
  --entry-point monitor_model \
  --trigger-http \
  --timeout 540s \
  --memory 512MB

# Grant permission to call Cloud Run
gcloud run services add-iam-policy-binding himas-prediction-service \
  --region us-central1 \
  --member "serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role roles/run.invoker
```

### 4. Set Up Automated Monitoring Schedule

```bash
# Get function URL
FUNCTION_URL=$(gcloud functions describe model-monitor \
  --region us-central1 \
  --gen2 \
  --format="value(serviceConfig.uri)")

# Create scheduler job (every 6 hours, EST timezone)
gcloud scheduler jobs create http model-monitor-6hourly \
  --location us-central1 \
  --schedule "0 */6 * * *" \
  --time-zone "America/New_York" \
  --uri "${FUNCTION_URL}" \
  --http-method POST \
  --oidc-service-account-email "${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --oidc-token-audience "${FUNCTION_URL}"
```

---

## Testing the Deployment

### 1. Health Check

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe himas-prediction-service \
  --region us-central1 \
  --format 'value(status.url)')

# Test health endpoint
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  ${SERVICE_URL}/health
```

**Expected:** `{"status":"healthy"}`

### 2. Make a Prediction

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "age_at_admission": 65,
    "los_icu_hours": 72,
    "los_icu_days": 3,
    "gender": "M",
    "insurance": "Medicare"
  }' \
  ${SERVICE_URL}/predict
```

**Expected:** `{"prediction": 0 or 1, "probability": 0.0-1.0}`

### 3. Test with Real Data

```bash
cd PoC/Model-Serving/testing
python3 test_deployed_model.py
```

**Output:** Tests 50 samples from BigQuery, reports accuracy metrics

### 4. Trigger Monitoring Manually

```bash
# Trigger Cloud Scheduler job
gcloud scheduler jobs run model-monitor-6hourly --location us-central1

# Or call function directly
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://us-central1-${PROJECT_ID}.cloudfunctions.net/model-monitor
```

---

## Triggering Retraining

### Automatic Retraining Scenarios

1. **Model Decay Detected**
   - Cloud Function finds accuracy < 85% or recall < 70%
   - Manual action: Push any change to GitHub to trigger full pipeline

2. **Code/Data Changes**
   - Any push to `main` branch automatically:
     - Retrains model on latest BigQuery data
     - Evaluates and validates
     - Deploys if quality gates pass

3. **Manual Retraining**
   ```bash
   # Option 1: Empty commit to trigger pipeline
   git commit --allow-empty -m "Trigger retraining"
   git push origin main
   
   # Option 2: Trigger Cloud Build directly
   gcloud builds submit --config cloudbuild.yaml
   ```

### Retraining Protection

The pipeline includes **automatic rollback protection**:

- **Metric validation:** New model must meet minimum thresholds
- **Model comparison:** New model must not be >2% worse than previous
- **Bias detection:** Must pass fairness checks
- **If any fail:** Pipeline stops, production keeps serving old model

**Location in Code:**
- Thresholds: `cloudbuild.yaml` lines 91-110 (`validate-metrics`)
- Comparison: `cloudbuild.yaml` lines 129-169 (`compare-with-previous`)
- Bias: `cloudbuild.yaml` lines 112-127 (`bias-detection`)

---

## Monitoring Dashboards

### 1. Model Performance Over Time (BigQuery)

**Console:** Cloud Console → BigQuery → SQL Workspace

**Query:**
```sql
SELECT 
  timestamp,
  samples_tested,
  ROUND(accuracy * 100, 2) as accuracy_pct,
  ROUND(recall * 100, 2) as recall_pct,
  ROUND(precision * 100, 2) as precision_pct
FROM `erudite-carving-472018-r5.production_monitoring.model_performance`
WHERE samples_tested > 0
ORDER BY timestamp DESC
```

**Shows:**
- Performance trends over time
- Model decay detection
- Historical baseline for comparison

### 2. Cloud Run Service Metrics

**Console:** Cloud Console → Cloud Run → himas-prediction-service → Metrics

**Metrics Available:**
- Request count (predictions per second)
- Request latency (p50, p95, p99)
- Container CPU utilization
- Container memory usage
- Error rate (4xx, 5xx responses)
- Active instances count

### 3. Cloud Function Monitoring

**Console:** Cloud Console → Cloud Functions → model-monitor → Metrics

**Metrics Available:**
- Invocation count (monitoring runs)
- Execution time
- Error rate
- Active instances

### 4. Cloud Scheduler Status

**Console:** Cloud Console → Cloud Scheduler

**Shows:**
- Schedule: `0 */6 * * *` (every 6 hours EST)
- Last run time
- Success/failure status
- Next scheduled run

### 5. Cloud Build History

**Console:** Cloud Console → Cloud Build → History

**Shows:**
- All pipeline runs (training + deployment)
- Duration of each step
- Success/failure status
- Build logs for debugging

---

## Key Endpoints

| Endpoint | Purpose | Authentication |
|----------|---------|----------------|
| `https://himas-prediction-service-erudite-carving-472018-r5.us-central1.run.app/health` | Health check | Required |
| `https://himas-prediction-service-erudite-carving-472018-r5.us-central1.run.app/predict` | Make predictions | Required |
| `https://us-central1-erudite-carving-472018-r5.cloudfunctions.net/model-monitor` | Monitoring function | Required |

**Authentication:**
```bash
# Get auth token
TOKEN=$(gcloud auth print-identity-token)

# Use in requests
curl -H "Authorization: Bearer $TOKEN" {endpoint}
```

---

## Model Artifacts Storage

All model artifacts are stored in GCS with timestamp-based organization:

```
gs://himas-mlops-models/
├── models/
│   └── {timestamp}/
│       ├── model.keras                    # Trained model
│       └── preprocessing/
│           ├── scaler.pkl                 # StandardScaler
│           ├── label_encoders.pkl         # Categorical encoders
│           └── feature_order.pkl          # Feature ordering
├── evaluation-results/
│   └── {timestamp}/
│       └── results/
│           └── evaluation_results_{ts}.json
├── mlflow-runs/
│   └── {timestamp}/
│       └── mlruns/                        # MLflow tracking
└── model-registry/
    ├── {timestamp}.json                   # Model card (metadata)
    └── latest.txt                         # Pointer to latest model
```

**Model Card Contents:**
- Model version and timestamp
- Performance metrics (accuracy, recall, precision, F1, ROC-AUC)
- Artifact locations (model file, preprocessing, results)
- Training configuration (framework, hospitals, hyperparameters)

---

## Configuration

### Monitoring Thresholds

**File:** `cloud-functions/model-monitor/main.py`

```python
ACCURACY_THRESHOLD = 0.85  # 85%
RECALL_THRESHOLD = 0.70    # 70%
NUM_TEST_SAMPLES = 100     # Samples to test per run
```

### Cloud Run Configuration

**File:** `cloudbuild.yaml` (deploy-cloudrun step)

```yaml
--memory=2Gi              # 2GB memory
--cpu=2                   # 2 CPU cores
--max-instances=10        # Scale up to 10
--min-instances=1         # Keep 1 always running
```

### Monitoring Schedule

**Modify schedule:**
```bash
gcloud scheduler jobs update http model-monitor-6hourly \
  --location us-central1 \
  --schedule "0 */4 * * *"  # Change to every 4 hours
```

**Cron format:** `minute hour day month weekday`
- Every 6 hours: `0 */6 * * *`
- Every 4 hours: `0 */4 * * *`
- Daily at 2 AM: `0 2 * * *`

---

## Replication Instructions

### Fresh Environment Setup (Video Demonstration)

**Step 1: Initial GCP Setup (5 min)**
```bash
# Authenticate
gcloud auth login

# Set project
gcloud config set project {your-project-id}

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com \
  cloudfunctions.googleapis.com cloudscheduler.googleapis.com \
  bigquery.googleapis.com artifactregistry.googleapis.com
```

**Step 2: Create Storage (2 min)**
```bash
# GCS bucket
gsutil mb -l us-central1 gs://himas-mlops-models

# BigQuery monitoring dataset + table
bq mk --dataset --location us-central1 erudite-carving-472018-r5:production_monitoring
bq mk --table erudite-carving-472018-r5:production_monitoring.model_performance \
  timestamp:TIMESTAMP,samples_tested:INTEGER,accuracy:FLOAT,recall:FLOAT,\
  precision:FLOAT,tp:INTEGER,fp:INTEGER,fn:INTEGER,tn:INTEGER
```

**Step 3: Connect GitHub Repo (3 min)**
```bash
# Clone repository
git clone https://github.com/manjushamg26/himas-mlops.git
cd himas-mlops

# Create Cloud Build trigger
gcloud builds triggers create github \
  --name himas-mlops-trigger \
  --repo-name himas-mlops \
  --repo-owner manjushamg26 \
  --branch-pattern ^main$ \
  --build-config cloudbuild.yaml
```

**Step 4: First Deployment (15-20 min)**
```bash
# Trigger pipeline by pushing to GitHub
# (or manually trigger)
gcloud builds submit --config cloudbuild.yaml

# Wait for completion - Cloud Build will:
# - Train model
# - Run quality checks
# - Deploy to Cloud Run automatically
```

**Step 5: Deploy Monitoring (5 min)**
```bash
cd PoC/Model-Serving/cloud-functions/model-monitor

# Deploy Cloud Function
gcloud functions deploy model-monitor \
  --gen2 --runtime python311 --region us-central1 \
  --source . --entry-point monitor_model \
  --trigger-http --timeout 540s --memory 512MB

# Grant permissions
gcloud run services add-iam-policy-binding himas-prediction-service \
  --region us-central1 \
  --member "serviceAccount:1089649594993-compute@developer.gserviceaccount.com" \
  --role roles/run.invoker

# Set up schedule
gcloud scheduler jobs create http model-monitor-6hourly \
  --location us-central1 \
  --schedule "0 */6 * * *" \
  --time-zone "America/New_York" \
  --uri "https://us-central1-erudite-carving-472018-r5.cloudfunctions.net/model-monitor" \
  --http-method POST \
  --oidc-service-account-email "1089649594993-compute@developer.gserviceaccount.com" \
  --oidc-token-audience "https://us-central1-erudite-carving-472018-r5.cloudfunctions.net/model-monitor"
```

**Step 6: Verify Deployment (2 min)**
```bash
# Test prediction
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"age_at_admission": 65, "los_icu_hours": 72}' \
  https://himas-prediction-service-erudite-carving-472018-r5.us-central1.run.app/predict

# Trigger monitoring
gcloud scheduler jobs run model-monitor-6hourly --location us-central1

# Check results in BigQuery
bq query --use_legacy_sql=false \
  "SELECT * FROM \`erudite-carving-472018-r5.production_monitoring.model_performance\` 
   ORDER BY timestamp DESC LIMIT 1"
```

**Total Time:** ~30 minutes for complete setup on fresh environment

---

## Monitoring Model Performance

### View Performance Trends

```sql
-- Performance over time
SELECT 
  timestamp,
  samples_tested,
  ROUND(accuracy, 4) as accuracy,
  ROUND(recall, 4) as recall,
  ROUND(precision, 4) as precision
FROM `erudite-carving-472018-r5.production_monitoring.model_performance`
WHERE samples_tested > 0
ORDER BY timestamp DESC
```

### Detect Model Degradation

```sql
-- Check if latest performance is worse than baseline
WITH latest AS (
  SELECT accuracy, recall
  FROM `erudite-carving-472018-r5.production_monitoring.model_performance`
  WHERE samples_tested > 0
  ORDER BY timestamp DESC LIMIT 1
),
baseline AS (
  SELECT AVG(accuracy) as avg_acc, AVG(recall) as avg_recall
  FROM `erudite-carving-472018-r5.production_monitoring.model_performance`
  WHERE samples_tested > 0
)
SELECT 
  latest.accuracy as current_accuracy,
  baseline.avg_acc as baseline_accuracy,
  latest.accuracy - baseline.avg_acc as accuracy_diff,
  CASE 
    WHEN latest.accuracy < 0.85 OR latest.recall < 0.70 THEN 'RETRAINING NEEDED'
    WHEN latest.accuracy < baseline.avg_acc - 0.02 THEN 'DEGRADATION DETECTED'
    ELSE 'OK'
  END as status
FROM latest, baseline
```

---

## Expected Performance

Based on validation runs:

| Metric | Expected Range | Latest Result |
|--------|---------------|---------------|
| Accuracy | 85-92% | 89.9% ✅ |
| Recall | 70-85% | 72.7% ✅ |
| Precision | 50-65% | 53.3% ✅ |
| ROC-AUC | 0.90-0.95 | 0.92 ✅ |

**Note:** Precision is intentionally lower - for mortality prediction, we prefer false alarms over missed deaths (high recall priority).

---

## Troubleshooting

### Issue: Cloud Run deployment fails

**Check logs:**
```bash
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=himas-prediction-service" \
  --limit 50 --order desc
```

**Common causes:**
- Model file not found: Check GCS bucket has latest model
- Import errors: Verify all Python dependencies in requirements.txt
- Memory issues: Increase --memory in deployment

### Issue: Monitoring returns 0 samples

**Check:**
```bash
# Verify test data exists
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) FROM \`erudite-carving-472018-r5.federated.hospital_a_data\` 
   WHERE data_split = 'test'"
```

**Fix:** Ensure BigQuery table has `data_split` column with 'test' values

### Issue: 401 Unauthorized errors

**Grant permissions:**
```bash
gcloud run services add-iam-policy-binding himas-prediction-service \
  --region us-central1 \
  --member "serviceAccount:1089649594993-compute@developer.gserviceaccount.com" \
  --role roles/run.invoker
```

### Issue: Cloud Build fails

**Check build logs:**
```bash
gcloud builds log {build-id}
```

**Common issues:**
- Missing GCS bucket: Create `gs://himas-mlops-models`
- BigQuery access: Grant Cloud Build service account BigQuery permissions
- Docker build fails: Check Dockerfile syntax

---

## Accessing Logs and Metrics

### Cloud Run Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=himas-prediction-service" \
  --limit 100 --format json
```

### Cloud Function Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=model-monitor" \
  --limit 100 --format json
```

### Cloud Build Logs
```bash
# List recent builds
gcloud builds list --limit 10

# View specific build
gcloud builds log {build-id}
```

---

## Cost Estimation

**Monthly costs (approximate):**

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run | 1 min instance + occasional traffic | ~$2-5 |
| Cloud Functions | 120 invocations/month (6hr schedule) | ~$0.50 |
| Cloud Scheduler | 1 job | $0.10 |
| GCS Storage | ~5GB model artifacts | $0.10 |
| BigQuery | Monitoring queries | $0.50 |
| Cloud Build | ~10 builds/month | $1-2 |
| **Total** | | **~$5-10/month** |

**Cost Optimization:**
- Set Cloud Run min-instances=0 (cold starts OK for demo)
- Reduce monitoring frequency (daily instead of 6-hourly)
- Delete old model versions from GCS

---

## Production Deployment Checklist

- [ ] GCS bucket created (`gs://himas-mlops-models`)
- [ ] BigQuery datasets created (federated, production_monitoring)
- [ ] Cloud Build trigger connected to GitHub
- [ ] Cloud Run service deployed and healthy
- [ ] Cloud Function deployed
- [ ] IAM permissions granted (Cloud Function → Cloud Run)
- [ ] Cloud Scheduler job created
- [ ] Test prediction successful
- [ ] Monitoring function tested manually
- [ ] Performance metrics visible in BigQuery
- [ ] Documentation complete
- [ ] Video demonstration recorded

---

## Support and Contact

**Team:** HIMAS MLOps Team  
**Repository:** https://github.com/manjushamg26/himas-mlops  
**GCP Project:** erudite-carving-472018-r5  

For issues or questions, check Cloud Build logs and monitoring dashboards first.