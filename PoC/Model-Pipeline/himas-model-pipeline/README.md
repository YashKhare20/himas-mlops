# **HIMAS Federated Learning Pipeline**

### Federated ICU Mortality Prediction using Flower + TensorFlow + BigQuery + MLflow

This project trains and evaluates a **federated deep learning model** across three virtual hospitals using the [Flower](https://flower.ai) framework.
It uses **BigQuery** as the data source, **MLflow** for experiment tracking, and **Docker Compose** to orchestrate the multi-container federation (Server, SuperNodes, Evaluator, MLflow server, and SuperLink).

---

## 🤖 **1. Architecture Overview**

**Components:**

* **SuperLink** — central router for federated coordination.
* **ServerApp** — orchestrates training and aggregates model updates.
* **SuperNodes** — simulate three client hospitals (A, B, C).
* **Evaluator** — evaluates the final global model on BigQuery test data.
* **MLflow** — tracks training metrics, models, and evaluation artifacts.

**Federation setup:**

```
┌──────────┐        ┌──────────┐
│ hospital_a │        │ hospital_b │
└─────┬──────┘        └──────┬─────┘
      │                     │
      ├─── flower-superlink ─────┬
      │                     │     │
  ┌───▼──────┐        ┌─────▼────┐
  │ hospital_c │        │ ServerApp │
  └────────────┘        └──────────┘
          │
      ┌───▼───┐
      │ MLflow│
      └───────┘
```

---

## ⚙️ **2. Initial Setup (one-time)**

### **A. Clone the repository**

```bash
git clone https://github.com/your-org/himas-model-pipeline.git
cd himas-model-pipeline/PoC/Model-Pipeline/himas-model-pipeline
```

### **B. Install prerequisites**

Ensure you have:

* **Python 3.11+**
* **Docker Desktop**
* **gcloud CLI** ([install guide](https://cloud.google.com/sdk/docs/install))

Then install Python dependencies from `pyproject.toml`:

```bash
pip install -e .
```

This will automatically install all dependencies declared under `[project.dependencies]` in `pyproject.toml`.

---

### **C. Environment Variables Setup**

A template environment file is provided as `.env.example`. Copy it and customize if needed:

```bash
cp .env.example .env
```

The `.env.example` file contains portable defaults:

```bash
# === GCP credentials for BigQuery ===
GOOGLE_CLOUD_PROJECT=erudite-carving-472018-r5

# === Optional tracking config ===
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=himas-federated
```

---

## 🌍 **3. Authenticate with GCP (no service account needed)**

We use **Application Default Credentials (ADC)** via your **Google user account** — no `key.json` is required.

Run this once on your system:

```bash
gcloud auth application-default login
```

This creates your credentials at:

```
C:\Users\%USERNAME%\AppData\Roaming\gcloud\application_default_credentials.json
```

---

## 🪄 **4. Configure Docker Authentication**

### Update in `docker-compose.yml` (already included)

Your local credentials are mounted automatically for all containers:

```yaml
volumes:
  - "C:/Users/%USERNAME%/AppData/Roaming/gcloud:/gcloud:ro"

environment:
  GOOGLE_APPLICATION_CREDENTIALS: /gcloud/application_default_credentials.json
  GOOGLE_CLOUD_PROJECT: erudite-carving-472018-r5
```

✅ This allows every container (server, supernodes, evaluator) to use your GCP user credentials for BigQuery access.

---

## 🚀 **5. Build and Run the Federated Pipeline**

### Step 1: Build containers

```bash
docker compose build
```

### Step 2: Launch the full federation

```bash
docker compose up
```

This spins up:

* `himas-superlink`
* `himas-server`
* `himas-supernode-a`
* `himas-supernode-b`
* `himas-supernode-c`
* `himas-mlflow`
* `himas-evaluator`

Each will start training automatically and log metrics to **MLflow**.

### Step 3: View training progress

Open [http://localhost:5000](http://localhost:5000) in your browser to access **MLflow UI**.

You’ll see:

* Separate runs for each client (`client-train-p0`, `client-eval-p1`, etc.)
* The federated session run (`server-session`)
* Final evaluation metrics (`evaluation`)

---

## 🧪 **6. Running Only Evaluation**

If you just want to evaluate the last trained model again:

```bash
docker compose run evaluator
```

Results:

* Markdown & JSON reports → `/evaluation_results`
* Plots → `/evaluation_results/figures`
* Logged automatically to MLflow under `himas-federated-eval`

---

## 🧠 **7. Modifying or Updating the Model**

### **Where the model is defined**

File:
`himas_model_pipeline/task.py`
→ Function:

```python
def load_model(input_dim: int) -> keras.Model:
```

Modify this function to:

* Change architecture (layers, units, dropout, etc.)
* Add new metrics or optimizers
* Include attention/transformer modules if needed

### **If you add or remove features**

* Update the feature lists in the same file:

  ```python
  NUMERICAL_FEATURES = [...]
  CATEGORICAL_FEATURES = [...]
  ```
* Ensure the **input dimension** matches in `load_model()` and `get_feature_dim()` logic.

---

### **If you change preprocessing**

* Edit preprocessing logic in `DataPreprocessor` class inside `task.py`.
* The federated clients (`client_app.py`) automatically pick up updated preprocessing.

---

### **If you add new evaluation metrics**

* Update `evaluate_model.py` inside:

  ```python
  def evaluate_hospital(self, hospital: str)
  ```
* Add the new metrics to the `metrics` dictionary and include them in the MLflow logs.

---

### **After model/preprocessing change**

1. Rebuild the Docker images:

   ```bash
   docker compose build
   ```

2. Re-run the training:

   ```bash
   docker compose up
   ```

3. Evaluate:

   ```bash
   docker compose run evaluator
   ```

4. Review MLflow dashboard for performance improvements.

---

## 🗂️ **8. Project Structure**

```
himas-model-pipeline/
│
├── himas_model_pipeline/
│   ├── client_app.py         # Flower Client (SuperNode)
│   ├── server_app.py         # Flower Server
│   ├── task.py               # Model + Preprocessing + BigQuery data loader
│   ├── __init__.py
│
├── evaluate_model.py         # Aggregated evaluation + plots + report
├── docker-compose.yml        # Multi-container federation orchestration
├── models/                   # Saved Keras models
├── evaluation_results/       # Evaluation reports and figures
├── mlruns/                   # MLflow experiment logs
├── mlflow.db                 # Local MLflow SQLite backend
├── pyproject.toml            # Dependency and build management
├── .env.example              # Environment variable template
└── README.md
```
---

## 💡 **9. Typical Workflow Summary**

| Step              | Command                                           | Description                    |
| ----------------- | ------------------------------------------------- | ------------------------------ |
| Authenticate once | `gcloud auth application-default login`           | Creates user ADC               |
| Build containers  | `docker compose build`                            | Prepare environment            |
| Start federation  | `docker compose up`                               | Train model across 3 hospitals |
| Evaluate model    | `docker compose run evaluator`                    | Generate reports & plots       |
| View logs         | `http://localhost:5000`                           | MLflow dashboard               |
| Modify model      | Edit `task.py`                                    | Update layers/features         |
| Retrain           | `docker compose up`                               | New global model               |
| Save outputs      | Auto-saved to `/models/` & `/evaluation_results/` | Versioned locally & in MLflow  |

