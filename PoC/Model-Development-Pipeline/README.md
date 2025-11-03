# 🧠 Model-Development-Pipeline

> **End-to-End Federated ML Pipeline** for hospital-distributed datasets.
> Combines BigQuery data export, per-hospital data splitting, and Flower-based federated training using Logistic Regression, MLP, and TabNet models.

---

## 📁 Repository Overview

```
Model-Development-Pipeline/
│
├── config.py                        # Global configuration (paths, models, rounds, metrics, etc.)
├── load_data.py                     # Exports curated/federated datasets from BigQuery
├── split_data.py                    # Splits per-hospital data into train/validation/test subsets
├── train_federated_flower.py        # Federated Learning with Flower (server & client)
├── artifacts/
│   ├── splits/                      # Train/validation/test CSVs for each hospital
│   ├── models_federated_true/       # Serialized model weights
│   └── reports_federated_true/      # Evaluation reports & JSON summaries
└── utils.py                         # Helper functions (feature selection, directory setup, etc.)
```

---

## ⚙️ 1. Environment Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Recommended `requirements.txt`:

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
protobuf<5,>=4.21.6
```
---

## 🧩 2. Configuration (config.py)

All runtime parameters and paths are centralized in `config.py`:

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
PROX_MU = 0.0        # FedProx coefficient (0 = FedAvg)

# Columns and metrics
TARGET_COL = "icu_mortality_label"
SPLIT_COL = "data_split"
EVAL_METRICS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall_sensitivity"]
THRESHOLD = 0.5
```

---

## ☁️ 3. Export Data from BigQuery

Run this only in the **BigQuery environment**:

```bash
python load_data.py
```

This script:

* Connects to the specified `PROJECT_ID`.
* Lists all tables in `curated_demo` and `federated_demo`.
* Exports each as both `.csv` and `.parquet` into `PoC/Data-Pipeline/data/`.

Example log:

```
=== DATASET: federated_demo ===
→ Exporting federated_demo.hospital_a_data
   Saved: PoC/Data-Pipeline/data/federated_demo/hospital_a_data.csv
```

---

## 🧱 4. Split Data by Hospital

```bash
python split_data.py
```

This:

* Reads each `hospital_X_data.csv`.
* Validates that columns `subject_id`, `data_split`, and `icu_mortality_label` exist.
* Verifies balanced train/validation/test splits.
* Saves files like:

```
artifacts/splits/
├── hospital_a_train.csv
├── hospital_a_validation.csv
├── hospital_a_test.csv
```

---

## 🤝 5. Run Federated Learning (Flower)

### Step 1 — Start Server

```bash
python train_federated_flower.py --role server --address 127.0.0.1:8080
```

### Step 2 — Start Clients (in separate terminals)

```bash
python train_federated_flower.py --role client --address 127.0.0.1:8080 --hospital hospital_a
python train_federated_flower.py --role client --address 127.0.0.1:8080 --hospital hospital_b
python train_federated_flower.py --role client --address 127.0.0.1:8080 --hospital hospital_c
```

Each client:

* Loads its hospital’s data.
* Trains models locally (`logreg`, `mlp`, `tabnet`).
* Sends weights back to the server for aggregation.
* Evaluates both per-model and ensemble metrics.

---

## 📊 6. Outputs and Reports

After all rounds complete:

```
artifacts/reports_federated_true/
├── logreg_client_metrics.csv
├── mlp_client_metrics.csv
├── tabnet_client_metrics.csv
├── ensemble_client_metrics.csv
├── logreg_summary.json
├── mlp_summary.json
├── tabnet_summary.json
└── ensemble_summary.json
```

Each `*_summary.json` contains:

```json
{
  "rows": 3,
  "macro": {
    "roc_auc": 0.92,
    "pr_auc": 0.87,
    "f1": 0.73,
    "accuracy": 0.82
  },
  "micro": {
    "roc_auc": 0.93,
    "pr_auc": 0.88,
    "f1": 0.75,
    "accuracy": 0.83
  }
}
```

---

## 🧠 7. Understanding Results

| File                    | Meaning                                              |
| ----------------------- | ---------------------------------------------------- |
| `*_client_metrics.csv`  | Metrics for each hospital (client-side test data).   |
| `*_summary.json`        | Aggregated macro/micro averages per model type.      |
| `ensemble_summary.json` | Ensemble (average of model predictions) performance. |

To interpret:

* `macro` → mean of all hospitals.
* `micro` → weighted average by sample count (`n`).
* High ROC-AUC and PR-AUC indicate strong discrimination.
* Compare per-hospital differences to understand site variance.

---

## 🔧 8. Troubleshooting

| Issue                                 | Fix                                                                 |
| ------------------------------------- | ------------------------------------------------------------------- |
| `protobuf` version conflict           | Use `protobuf<5` for Flower, `>=6` for BigQuery (or separate envs). |
| `missing column: icu_mortality_label` | Verify target name in `config.py`.                                  |
| `TypeError: ArrayDataset32`           | Use Python 3.10–3.12 and scikit-learn 1.5+.                         |
| TabNet warnings about `spmatrix`      | Safe to ignore (SciPy 2.0 deprecation).                             |

---

## 🧾 License & Credits

Developed as part of a federated learning proof-of-concept (PoC) demonstrating:

* Secure multi-hospital model training
* End-to-end pipeline automation
* Model aggregation, evaluation, and interpretability

Built with:

* **Flower** (Federated Learning Framework)
* **PyTorch** & **TabNet**
* **Scikit-learn** for traditional models
* **Google BigQuery** for distributed data sourcing