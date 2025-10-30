
import os, json, socket
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import mlflow
import flwr as fl
import logging, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
for name in ["flwr", "flwr.client", "flwr.server"]:
    logging.getLogger(name).setLevel(logging.ERROR)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "federated-learning"))

DATA_DIR   = Path(os.getenv("DATA_DIR", "/app/data"))
HOSPITAL_ID= os.getenv("HOSPITAL_ID", "1")
SERVER_ADDR= os.getenv("SERVER_ADDRESS", "flower-server:8080")
RUN_NAME   = f"client_h{HOSPITAL_ID}"
DP_SIGMA   = float(os.getenv("DP_SIGMA", "0.0"))  # e.g., 0.01 for small noise

meta = json.loads((DATA_DIR/"feature_list.json").read_text())
FEATURES = meta["features"]
LABEL    = meta["label"]
CLASSES  = np.array(meta["classes"], dtype=int)

df = pd.read_csv(DATA_DIR/f"hosp{HOSPITAL_ID}.csv")
X = df[FEATURES].values.astype(np.float64)
y = df[LABEL].values.astype(int)

# local split for quick validation
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model (no scaler here; data is pre-scaled in generator)
clf = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", random_state=0)

# Initialize coef_ by calling partial_fit once
clf.partial_fit(X_tr[: min(32, len(X_tr))], y_tr[: min(32, len(y_tr))], classes=CLASSES)

def get_params_from_model(model: SGDClassifier) -> list[np.ndarray]:
    return [model.coef_.ravel().copy(), model.intercept_.copy()]

def set_params_to_model(model: SGDClassifier, params: list[np.ndarray]):
    coef_flat, intercept = params
    n_classes = len(CLASSES)
    n_features = len(FEATURES)
    model.coef_ = coef_flat.reshape(n_classes if n_classes>2 else 1, n_features).copy()
    model.intercept_ = intercept.copy()
    model.classes_ = CLASSES.copy()

class HospitalClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_params_from_model(clf)

    def fit(self, parameters, config):
        set_params_to_model(clf, parameters)
        local_epochs = int(config.get("local_epochs", 3))
        for _ in range(local_epochs):
            clf.partial_fit(X_tr, y_tr, classes=CLASSES)
        # Eval locally
        yhat_proba = clf.predict_proba(X_va)[:, 1] if len(CLASSES)==2 else None
        yhat = clf.predict(X_va)
        acc = accuracy_score(y_va, yhat)
        metrics = {"acc": float(acc)}
        if yhat_proba is not None:
            try:
                auc = roc_auc_score(y_va, yhat_proba)
                ll  = log_loss(y_va, yhat_proba)
                metrics.update({"auc": float(auc), "logloss": float(ll)})
            except Exception:
                pass

        # MLflow log
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        with mlflow.start_run(run_name=RUN_NAME, nested=False):
            for k,v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_param("hospital_id", HOSPITAL_ID)
            mlflow.log_param("dp_sigma", DP_SIGMA)

        # Simple DP: add Gaussian noise to returned parameters
        params = get_params_from_model(clf)
        if DP_SIGMA > 0.0:
            noisy = []
            for arr in params:
                scale = max(1e-6, float(np.linalg.norm(arr)) / arr.size)
                noisy.append(arr + np.random.normal(loc=0.0, scale=DP_SIGMA*scale, size=arr.shape))
            params = noisy

        return params, len(X_tr), metrics

    def evaluate(self, parameters, config):
        set_params_to_model(clf, parameters)
        yhat = clf.predict(X_va)
        acc = accuracy_score(y_va, yhat)
        yhat_proba = clf.predict_proba(X_va)[:, 1] if len(CLASSES)==2 else None
        loss = float(1.0 - acc)
        if yhat_proba is not None:
            try:
                ll = log_loss(y_va, yhat_proba)
                loss = float(ll)
            except Exception:
                pass
        return loss, len(X_va), {"acc": float(acc)}

if __name__ == "__main__":
    print(f"Starting client H{HOSPITAL_ID} -> {SERVER_ADDR}")
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=HospitalClient())
