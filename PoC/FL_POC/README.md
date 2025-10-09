# Federated ICU-LOS Prediction (POC) — README

> **Goal :** build a small, end-to-end **federated learning** demo that predicts whether a patient’s **ICU length-of-stay (LOS)** will be **≥ 48 hours** using only the **public MIMIC-IV demo** data, with three “hospitals” training locally and a central server aggregating models.

---

## 1) What we’re solving (and why federated?)

Hospitals want an early signal that an ICU stay will be **prolonged** (≥48h). That helps with **bed planning, staffing, and step-down vs escalation**. But **data can’t be pooled** across sites.

**Federated learning (FL)** lets sites train **locally** and share **model updates only**. Our POC spins up **3 hospitals** as Docker containers that each train on a **non-overlapping slice** of the MIMIC demo; a **Flower** server aggregates with **FedAvg**. **MLflow** tracks experiments and metrics.

---

## 2) Repo layout (POC part)

```
poc/
├─ data/
│  ├─ generate_samples.py        # builds the tabular datasets for the POC
│  ├─ config.json                # configuration for data builder defaults (e.g., MIMIC base path)
│  ├─ hosp1.csv / hosp2.csv / hosp3.csv
│  ├─ global_test.csv
│  ├─ feature_list.json          # list of model features
│  └─ scaler_info.json           # min/max used to scale features
├─ clients/
│  ├─ Dockerfile.client
│  └─ train_client.py            # FL client: loads hospX.csv, trains locally
├─ server/
│  ├─ Dockerfile.server
│  ├─ server.py                  # Flower server (FedAvg, optional DP)
│  └─ evaluate_global_model.py   # evaluate final model on global_test.csv
├─ artifacts/                    # models and other outputs (mounted volume)
├─ mlruns/                       # MLflow tracking data (mounted volume)
├─ docker-compose.yml
└─ requirements.txt
```

> **Note on `poc/data/config.json`**
> This file provides defaults for the **data generation step** (e.g., the input MIMIC demo path). Command-line flags still take precedence.

---

## 3) The dataset we use

**Source:** *MIMIC-IV Clinical Database — Demo v2.2* (public).
You should have a folder that **directly contains** `hosp/` and `icu/`, e.g.

```
.../mimic-iv-clinical-database-demo-2.2/mimic-iv-clinical-database-demo-2.2/
  ├─ hosp/
  │   ├─ patients.csv.gz
  │   ├─ admissions.csv.gz
  │   ├─ labevents.csv.gz
  │   └─ d_labitems.csv.gz
  └─ icu/
      └─ icustays.csv.gz
```

**Tables we actually use**

* `icu/icustays.csv.gz` → ICU **stay_id**, **intime/outtime**, **first_careunit**
* `hosp/admissions.csv.gz` → **hadm_id**, **admittime/dischtime**
* `hosp/patients.csv.gz` → **subject_id**, **gender**, **anchor_age**
* `hosp/labevents.csv.gz` → labs (**itemid**, **charttime**, **valuenum**, **valueuom**)

> We **do not** use identifiers or free text. Everything is already de-identified in the demo.

---

## 4) How `generate_samples.py` builds the training data

Run from `poc/`:

```bash
python data/generate_samples.py
```

> **Configuration:**
> `generate_samples.py` will read defaults from `poc/data/config.json` (e.g., `mimic_base`) if present.
> You can **override** any default with CLI flags (e.g., `--mimic_base "<path>"`).

### 4.1 Cohort & label

* Join `icustays` → `admissions` → `patients` on `(subject_id, hadm_id)`
* Compute **ICU_LOS (hours)** = `(outtime - intime)`
* Define **label**: `target_prolonged_icu = 1` if `ICU_LOS ≥ 48`, else `0`

### 4.2 Features

1. **Demographics / unit**

* `anchor_age` (approximate age)
* `gender_*` (one-hot; baseline dropped)
* `first_careunit_*` (one-hot; baseline dropped)

2. **Labs inside the ICU window**

* Filter `labevents` to **charttime ∈ [intime, outtime]**
* For each `(stay_id, itemid)` compute the **mean** of numeric `valuenum`
* Pivot wide → columns named **`lab_mean_<itemid>`**

3. **Cleaning & scaling**

* One-hot encode categoricals (drop one level)
* Median impute missing numeric values
* **Min–max scale** all features to **[0, 1]**
  (min/max stored in `scaler_info.json` for reproducibility)

### 4.3 Splits (to simulate 3 hospitals)

* **Global hold-out test (~20%)**: select subjects where `subject_id % 10 ∈ {0,1}`
* Remaining subjects are assigned to hospitals deterministically:

  * `hosp1`: `subject_id % 3 == 0`
  * `hosp2`: `subject_id % 3 == 1`
  * `hosp3`: `subject_id % 3 == 2`

### 4.4 Files produced (under `poc/data/`)

* `hosp1.csv`, `hosp2.csv`, `hosp3.csv` — local training tables (one hospital each)
* `global_test.csv` — evaluation set not seen by any client
* `feature_list.json` — exact list of features used
* `scaler_info.json` — min/max per feature used for scaling

---

## 5) Data dictionary (what’s in those CSVs?)

All four CSVs share the same schema.

**Identifier & label columns**

* `subject_id` *(int)*: anonymized patient id
* `hadm_id` *(int)*: hospital admission id
* `stay_id` *(int)*: ICU stay id
* `target_prolonged_icu` *(0/1)*: **label** (1 if `ICU_LOS ≥ 48h`)
* `icu_los_hrs` *(float)*: actual ICU LOS in hours (not used as a predictor)

**Demographic & unit features**

* `anchor_age` *(float in [0,1] after scaling)*: approximate age
* `gender_*` *(0/1)*: one-hot indicator(s); baseline gender is dropped
* `first_careunit_*` *(0/1)*: one-hot unit indicators; baseline unit is dropped

**Lab features**

* `lab_mean_<itemid>` *(float in [0,1])*: mean numeric result for `itemid` **during the ICU stay**

  * To see human-readable names, join with `hosp/d_labitems.csv.gz` on `itemid`:

    ```python
    import pandas as pd, json, re
    feat = json.load(open("poc/data/feature_list.json"))["features"]
    lab_ids = [int(x.split("_")[-1]) for x in feat if x.startswith("lab_mean_")]
    dlab = pd.read_csv("<MIMIC_BASE>/hosp/d_labitems.csv.gz", low_memory=False)
    print(dlab[dlab["itemid"].isin(lab_ids)][["itemid","label","fluid","category","loinc_code"]])
    ```

> Exact feature names for your run are listed in **`poc/data/feature_list.json`**.

---

## 6) Federated pipeline (what runs where)

```
         ┌───────────┐             rounds of    ┌───────────┐
         │  Client 1 │──gradients/weights──►    │           │
         └───────────┘                         ┌▶│ Flower    │
         ┌───────────┐     (no raw data)       │ │  Server  │──global model
         │  Client 2 │──gradients/weights──►   │ └───────────┘
         └───────────┘                         │
         ┌───────────┐                         │ FedAvg
         │  Client 3 │──gradients/weights──►   │ (optional DP)
         └───────────┘                         └──────────────
```

* **Flower server** (`server/server.py`): starts FL on port `8080`, strategy = **FedAvg**
  (can enable differential privacy via config/env, if desired)
* **Clients** (`clients/train_client.py`): each loads `hospX.csv`, splits locally into train/val, trains for a few epochs, returns updates
* **MLflow**: tracks **per-round metrics** (loss, AUROC/accuracy if implemented) and **artifacts** (models)

---

## 7) How to run

### 7.1 Generate data (already done if you see CSVs under `poc/data`)

```bash
cd PoC/FL_POC
python data/generate_samples.py
```

*By default, the script reads `poc/data/config.json` for its **MIMIC base path** and other defaults. Use `--mimic_base "<path>"` to override.*

### 7.2 Start the POC with Docker

> Ensure **Docker Desktop** is running and WSL integration is enabled on Windows.

```bash
cd PoC/FL_POC
docker compose up --build
```

* Flower server and 3 clients will start; MLflow UI is at **[http://localhost:5000](http://localhost:5000)**
* Follow server logs:

  ```bash
  docker compose logs -f flower-server
  ```

### 7.3 Evaluate the final global model

If your compose setup writes the aggregated model to `artifacts/`:

```bash
# inside repo (optionally within the server container)
python server/evaluate_global_model.py
```

This script loads `global_test.csv`, applies the saved scaler, and prints metrics.

---

## 8) Interpreting results (what to expect)

* **Training logs** per client show local loss/metric per round.
* **MLflow** shows experiment runs, params (rounds, lr, batch size), metrics per round, and stored artifacts.
* **Evaluation** on `global_test.csv` reflects generalization to a held-out population.

This is a **pedagogical** pipeline; don’t expect clinical-grade performance from the tiny demo.

---

## 9) Design choices (short rationale)

* **Label = ICU LOS ≥ 48h** → clinically meaningful threshold; binary, simple to demo.
* **Deterministic splits** by `subject_id` modulo → reproducible and prevents leakage across hospitals.
* **ICU-window lab means** → captures early lab signal without time-series modeling; fast to compute.
* **Min–max scaling** with saved `scaler_info.json` → consistent transforms during training/eval.
* **Flower + MLflow** → clean separation of orchestration (FL) and experiment tracking (metrics/artifacts).

---

## 10) Common pitfalls & quick fixes

* **“No curated parquet found and demo base not auto-detected”**
  Pass `--mimic_base "<path that directly contains hosp/ and icu/>"` to `generate_samples.py`.
  *(If `poc/data/config.json` contains `"mimic_base"`, that will be used unless you override it via CLI.)*

* **Docker compose can’t start; named-pipe error on Windows**
  Start **Docker Desktop**, enable **WSL2** integration, then re-run `docker compose up`.

* **Files ended up in `poc/poc/data`**
  We fixed the script to always write into `poc/data`. If you have the old files, move them into `poc/data`.

---

## 11) How to extend this POC (ideas)

* Add more features: procedures, medications, microbiology, vitals (if available), comorbidity scores.
* Replace lab means with **time-aware** features (early 24h trends).
* Try stronger models (calibrated tree ensembles, shallow MLP) and compare sites.
* Turn on **differential privacy** noise at clients; study the privacy–utility trade-off.
* Add **fairness** slices (age bands, care units) in evaluation.

---

## 12) How this POC works end-to-end

### 12.1 Big picture

```
              ┌──────────────┐                 ┌───────────────────┐
 MIMIC demo → │ Data builder │  ➜  CSV/JSON →  │  3 client sites   │
 (hosp/,icu/) └──────────────┘                 │ (H1,H2,H3 in Docker)
                                                └─────────┬─────────┘
                                                          │ model updates (no raw data)
                                                          ▼
                                                   ┌──────────────┐
                                                   │ Flower server│  FedAvg rounds
                                                   └──────┬───────┘
                                                          │ global weights
                                                          ▼
                                                ┌───────────────────┐
                                                │  Global evaluator │  → metrics
                                                └───────────────────┘

Everything (logs, models, metrics) is tracked in **MLflow**; files persist on host via volumes.
```

---

### 12.2 Data: what we take and how we prepare it

1. **Source**: MIMIC-IV *demo* tables (`icu/icustays.csv.gz`, `hosp/{admissions,patients,labevents,d_labitems}.csv.gz`).
2. **Label**: `target_prolonged_icu = 1` if `(outtime - intime) ≥ 48h`, else `0`.
3. **Features**:

   * Demographics/unit: `anchor_age`, one-hot `gender_*`, one-hot `first_careunit_*`
   * Labs inside the ICU window: for each `(stay_id,itemid)`, mean of numeric `valuenum` → `lab_mean_<itemid>`
   * Missing numeric imputed (median); **min–max scaling** to `[0,1]`
4. **Deterministic splits** (no leakage):

   * **Global hold-out**: `subject_id % 10 ∈ {0,1}` → `data/global_test.csv`
   * Remainder → **H1/H2/H3** by `subject_id % 3`
5. **Outputs (under `poc/data/`)**:

   * `hosp1.csv`, `hosp2.csv`, `hosp3.csv`
   * `global_test.csv`
   * `feature_list.json` (columns used), `scaler_info.json` (min/max per feature)

> **Configuration:** the data builder reads defaults (e.g., `mimic_base`) from `poc/data/config.json`, which can be overridden via CLI flags.

---

### 12.3 Orchestration: what Docker Compose brings up

Compose starts four services on a private network:

* **mlflow**: experiment tracker/UI on `http://localhost:5000`

  * Backend store → `./mlruns/` (host volume)
  * Artifacts (saved models, confusion matrices, etc.) → `./artifacts/` (host volume)
* **flower-server**: FL coordinator on `:8080` (exposed to clients only)
* **client1/2/3**: one container per “hospital”

Health/dependency rules ensure the clients wait for MLflow/Flower to be ready. When the configured number of rounds finishes, **clients and server exit with code 0**; MLflow can keep running.

---

### 12.4 Training lifecycle (each run)

1. **Server boot** (`server/server.py`)

   * Creates/uses MLflow experiment (e.g., `federated-learning`)
   * Configures **FedAvg** strategy and number of rounds (`NUM_ROUNDS`)
   * Optional toggles (e.g., DP) can be wired via env/flags

2. **Client boot** (`clients/train_client.py`)

   * Reads env: `DATA_DIR`, `HOSPITAL_ID`, `SERVER_ADDRESS`, `MLFLOW_TRACKING_URI`
   * Loads its own CSV (`hosp{H}.csv`), applies the saved feature order and scaler
   * Builds a **simple classifier** (see script; e.g., a small sklearn/torch model)
   * Registers with Flower; opens a client-side MLflow run (e.g., `client_h{H}`)

3. **Federated rounds**

   * **Server → Clients**: send current global weights + training config
   * **Each client**: trains locally (mini-epochs on its data), computes local metrics, **sends back only model parameters/updates** (no raw rows)
   * **Server**: aggregates via **FedAvg**, logs a `server_round_{k}` MLflow run with aggregated metrics
   * Repeat for `NUM_ROUNDS`

4. **Global evaluation**

   * After the final aggregation, the server writes the **final global model** to `./artifacts/`
   * `server/evaluate_global_model.py` (manual or automated) loads:

     * saved global model
     * `data/global_test.csv`
     * `feature_list.json` + `scaler_info.json`
   * Computes metrics (e.g., accuracy/AUROC/PR, confusion matrix) and logs them to MLflow (run name like `global_eval`)

---

### 12.5 Why we keep a **global hold-out test**

* It’s an **unbiased** estimate of performance on a population **not used by any site**.
* Detects **overfitting** to a single hospital’s distribution or to the FedAvg process.
* Simulates a light form of **external validation** (generalization beyond a training site).

---

### 12.6 Where things are saved

* **MLflow backend store**: `poc/mlruns/`
  (experiments, runs, params, metrics, run metadata)
* **Artifacts**: `poc/artifacts/`
  (saved global models per round/final, plots, confusion matrix CSVs, etc.)
* **Prepared data**: `poc/data/`
  (hosp*.csv, global_test.csv, feature/scaler JSONs, `config.json` for data builder defaults)

Because these are **host-mounted volumes**, data persists across container rebuilds.

---

### 12.7 What each tool’s role is

* **Docker**: isolated, reproducible environments for the server and each “hospital”; no local Python/setup fuss.
* **Docker Compose**: one-command bring-up, service wiring (ports, networks, volumes), startup ordering/health.
* **Flower**: the **federated learning framework**; handles client discovery, orchestration, communication, and **FedAvg** aggregation.
* **MLflow**: experiment tracking + artifact store + web UI; makes runs comparable and auditable.
* **`generate_samples.py`**: deterministic data fabrication from MIMIC demo into site-specific CSVs and a shared scaler/feature manifest; reads defaults from `poc/data/config.json`.
* **Clients (`train_client.py`)**: local trainers; **never** ship raw data, only model updates + metrics.
* **Server (`server.py`)**: strategy/config, global aggregation, optional per-round eval, model checkpointing.
* **Evaluator (`evaluate_global_model.py`)**: loads the final global model and reports test metrics on the **global** hold-out.

---

### 12.8 Changing the knobs

* **Data input path**: set `"mimic_base"` in `poc/data/config.json` (used by `generate_samples.py`), or override with `--mimic_base "<path>"`.
* **Rounds**: set `NUM_ROUNDS` via environment/compose or server code.
* **Model/optimizer**: edit `clients/train_client.py`.
* **Privacy**: wire DP/noise/clipping in the client before sending updates.
* **Sites**: add another client service and another `hosp*.csv` split.
* **Tracking**: add more metrics/plots; MLflow will version everything automatically.

---
