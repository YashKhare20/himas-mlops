# 🧠 Model-Development-Pipeline

> **End-to-End Federated ML Pipeline** for multi-hospital datasets.
> This project integrates **BigQuery data export**, **per-hospital dataset splitting**, and **Flower-based Federated Learning** using **Logistic Regression**, **MLP**, and **TabNet** models — with full **MLflow** experiment tracking.

---

## 📁 Repository Structure

```
PoC/
└── Model-Development-Pipeline/
    ├── config.py                        # Global configuration for data paths, model setup, and metrics
    ├── load_data.py                     # Exports tables from BigQuery into local CSV/Parquet
    ├── split_data.py                    # Splits hospital datasets into train/validation/test
    ├── train_federated_flower.py        # Flower server/client training logic with MLflow tracking
    ├── utils.py                         # Helper functions (directory creation, feature selection, etc.)
    └── artifacts/
        ├── splits/                      # Train/validation/test splits per hospital
        ├── models_federated_true/       # Stored model weights (per model)
        ├── reports_federated_true/      # Metrics, summaries, and evaluation reports
        └── mlruns/                      # MLflow tracking logs (auto-generated)
```

---

## ⚙️ 1. Environment Setup

Run all commands from your project root:

```bash
cd C:\College\MLOPS\himas-mlops
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r PoC/Model-Development-Pipeline/requirements.txt
```

### ✅ Recommended `requirements.txt`

```text
flwr==1.22.0
scikit-learn==1.5.2
torch==2.9.0
pytorch-tabnet==4.1.0
numpy>=1.24
pandas>=2.2
matplotlib
pandas-gbq
google-cloud-bigquery
mlflow>=2.14.0
protobuf<5,>=4.21.6
```

---

## 🧩 2. Configuration (`config.py`)

All runtime paths and model parameters are set here:

```python
# BigQuery
PROJECT_ID = "erudite-carving-472018-r5"
DATASETS = ["curated_demo", "federated_demo"]

# Paths
DATA_ROOT = "PoC/Data-Pipeline/data"
FED_DATA_ROOT = f"{DATA_ROOT}/federated_demo"
SPLIT_OUT_DIR = "PoC/Model-Development-Pipeline/artifacts/splits"
MODEL_DIR = "PoC/Model-Development-Pipeline/artifacts/models_federated_true"
REPORT_DIR = "PoC/Model-Development-Pipeline/artifacts/reports_federated_true"

# Federated Learning setup
HOSPITALS = {"hospital_a": "a", "hospital_b": "b", "hospital_c": "c"}
MODELS = ["logreg", "mlp", "tabnet"]
ROUNDS_PER_MODEL = 3
PROX_MU = 0.0  # 0 = FedAvg, >0 = FedProx

# Columns and evaluation
TARGET_COL = "icu_mortality_label"
SPLIT_COL = "data_split"
EVAL_METRICS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall_sensitivity"]
THRESHOLD = 0.5

# MLflow setup
MLFLOW_TRACKING_URI = "file:./PoC/Model-Development-Pipeline/artifacts/mlruns"
MLFLOW_EXPERIMENT_PREFIX = "Federated_Experiment_"
LOG_MODE = "all"  # ('all' = CSV + JSON logs; 'summary' = only summary logs)
```

---

## ☁️ 3. Export Data from BigQuery

Run **once** to export the raw data from your BigQuery project:

```bash
python PoC/Model-Development-Pipeline/load_data.py
```

✅ **What this does**

* Connects to your BigQuery project (`PROJECT_ID`).
* Lists all tables under the datasets (`curated_demo`, `federated_demo`).
* Exports each as `.csv` and `.parquet` into:

```
PoC/Data-Pipeline/data/federated_demo/
```

**Example output**

```
=== DATASET: federated_demo ===
→ Exporting federated_demo.hospital_a_data
Saved: PoC/Data-Pipeline/data/federated_demo/hospital_a_data.csv
```

---

## 🧱 4. Split Data by Hospital

Next, run:

```bash
python PoC/Model-Development-Pipeline/split_data.py
```

✅ **What this does**

* Validates that each hospital dataset contains `subject_id`, `data_split`, and `icu_mortality_label`.
* Ensures each split has both class labels.
* Writes per-split outputs to:

```
PoC/Model-Development-Pipeline/artifacts/splits/
├── hospital_a_train.csv
├── hospital_a_validation.csv
└── hospital_a_test.csv
```

---

## 🔥 5. Start MLflow Tracking UI

Open a **new terminal** (keep your environment activated):

```bash
mlflow ui --port 5000 --backend-store-uri PoC/Model-Development-Pipeline/artifacts/mlruns
```

Then visit:
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

Keep this terminal running while you perform training — all experiments will be logged here automatically.

---

## 🤝 6. Run Federated Learning (Flower + MLflow)

Federated training requires **1 server** and **3 clients** (one per hospital).

### 🖥️ Step 1 — Start Server

```bash
python PoC/Model-Development-Pipeline/train_federated_flower.py --role server --address 127.0.0.1:8080
```

### 🏥 Step 2 — Start Clients (each in a separate terminal)

```bash
python PoC/Model-Development-Pipeline/train_federated_flower.py --role client --address 127.0.0.1:8080 --hospital hospital_a
python PoC/Model-Development-Pipeline/train_federated_flower.py --role client --address 127.0.0.1:8080 --hospital hospital_b
python PoC/Model-Development-Pipeline/train_federated_flower.py --role client --address 127.0.0.1:8080 --hospital hospital_c
```

✅ **Each client:**

* Loads its local data.
* Trains `logreg`, `mlp`, and `tabnet` models.
* Sends model weights to the server for aggregation.
* Evaluates locally on its own test data.
* Logs all metrics to **MLflow** and saves artifacts to `/reports_federated_true`.

---

## 📊 7. Outputs & Reports

After training finishes, check:

```
PoC/Model-Development-Pipeline/artifacts/reports_federated_true/
├── logreg_client_metrics.csv
├── mlp_client_metrics.csv
├── tabnet_client_metrics.csv
├── ensemble_client_metrics.csv
├── logreg_summary.json
├── mlp_summary.json
├── tabnet_summary.json
└── ensemble_summary.json
```

Example content:

```json
{
  "rows": 3,
  "macro": {"roc_auc": 0.92, "pr_auc": 0.87, "f1": 0.73, "accuracy": 0.82},
  "micro": {"roc_auc": 0.93, "pr_auc": 0.88, "f1": 0.75, "accuracy": 0.83}
}
```

---

## 🧠 8. Understanding Results

| File                    | Description                         |
| ----------------------- | ----------------------------------- |
| `*_client_metrics.csv`  | Client-level metrics (per hospital) |
| `*_summary.json`        | Aggregated macro/micro averages     |
| `ensemble_summary.json` | Combined model ensemble metrics     |

**Interpretation:**

* **Macro** → mean across hospitals.
* **Micro** → weighted average by number of samples.
* **ROC-AUC / PR-AUC** → quality of model discrimination.
* **Compare hospitals** to assess cross-site performance consistency.

---

## 📈 9. MLflow Experiment Tracking

All metrics, parameters, and artifacts are logged automatically to:

```
PoC/Model-Development-Pipeline/artifacts/mlruns/
```

View via the MLflow UI (`http://127.0.0.1:5000`).

You’ll see experiments:

```
Federated_Experiment_logreg
Federated_Experiment_mlp
Federated_Experiment_tabnet
Federated_Experiment_ensemble
```

Each run contains:

* Per-hospital metrics
* Model parameters
* JSON summaries
* Global model weights (if saving is enabled)

To log to a **remote MLflow server**, update in `config.py`:

```python
MLFLOW_TRACKING_URI = "http://<mlflow-server-host>:5000"
```

---

## 🧹 10. Optional Cleanup

To restart with a clean experiment space:

```bash
rm -rf PoC/Model-Development-Pipeline/artifacts/mlruns
rm -rf PoC/Model-Development-Pipeline/artifacts/reports_federated_true
```

---

## ⚙️ Recommended Run Order

| Step | Description               | Command                                                                                                  |
| ---- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1    | Activate environment      | `.\.venv\Scripts\activate`                                                                               |
| 2    | Install dependencies      | `pip install -r PoC/Model-Development-Pipeline/requirements.txt`                                         |
| 3    | Export data from BigQuery | `python PoC/Model-Development-Pipeline/load_data.py`                                                     |
| 4    | Split datasets            | `python PoC/Model-Development-Pipeline/split_data.py`                                                    |
| 5    | Start MLflow UI           | `mlflow ui --port 5000 --backend-store-uri PoC/Model-Development-Pipeline/artifacts/mlruns`              |
| 6    | Run Flower server         | `python PoC/Model-Development-Pipeline/train_federated_flower.py --role server --address 127.0.0.1:8080` |
| 7    | Run Flower clients        | (3 terminals) — `hospital_a`, `hospital_b`, `hospital_c`                                                 |
| 8    | View results in MLflow    | [http://127.0.0.1:5000](http://127.0.0.1:5000)                                                           |

---

## 🧾 License & Credits

Developed as part of a **Federated Learning Proof of Concept (PoC)** demonstrating:

* Secure multi-site model training (without centralizing data)
* End-to-end automated pipelines
* Model evaluation, aggregation, and MLflow-based tracking

**Built with:**

* **Flower** — Federated Learning Framework
* **PyTorch + TabNet** — Deep learning models
* **Scikit-learn** — Classical ML models
* **Google BigQuery** — Distributed data source
* **MLflow** — Experiment management and visualization
