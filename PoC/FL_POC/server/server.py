
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import mlflow
import flwr as fl
import logging, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
for name in ["flwr", "flwr.client", "flwr.server"]:
    logging.getLogger(name).setLevel(logging.ERROR)
    # server/server.py

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "federated-learning"))  # creates if missing



DATA_DIR   = Path(os.getenv("DATA_DIR", "/app/data"))
ART_DIR    = Path(os.getenv("ART_DIR", "/app/artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)

meta = json.loads((DATA_DIR/"feature_list.json").read_text())
FEATURES = meta["features"]; LABEL = meta["label"]; CLASSES = np.array(meta["classes"], dtype=int)

test_df = pd.read_csv(DATA_DIR/"global_test.csv")
X_te = test_df[FEATURES].values.astype(np.float64)
y_te = test_df[LABEL].values.astype(int)

# Build a model with correct shape
clf = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", random_state=0)
# initialize shape
clf.partial_fit(X_te[: min(32, len(X_te))], y_te[: min(32, len(y_te))], classes=CLASSES)

def get_params_from_model(model):
    return [model.coef_.ravel().copy(), model.intercept_.copy()]

def set_params_to_model(model, params):
    coef_flat, intercept = params
    n_classes = len(CLASSES); n_features = len(FEATURES)
    model.coef_ = coef_flat.reshape(n_classes if n_classes>2 else 1, n_features).copy()
    model.intercept_ = intercept.copy()
    model.classes_ = CLASSES.copy()

def save_global(params):
    np.savez(ART_DIR/"global_model.npz", coef=params[0], intercept=params[1], features=np.array(FEATURES, dtype=object))

def evaluate_global(params):
    set_params_to_model(clf, params)
    yhat = clf.predict(X_te)
    acc  = accuracy_score(y_te, yhat)
    loss = float(1.0 - acc)
    metrics = {"acc": float(acc)}
    try:
        proba = clf.predict_proba(X_te)[:,1]
        auc   = roc_auc_score(y_te, proba)
        ll    = log_loss(y_te, proba)
        metrics.update({"auc": float(auc), "logloss": float(ll)})
        loss = float(ll)
    except Exception:
        pass
    return loss, metrics

def get_evaluate_fn():
    def evaluate(server_round, parameters, config):
        loss, metrics = evaluate_global(parameters)
        # Log to MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        with mlflow.start_run(run_name=f"server_round_{server_round}", nested=False):
            for k,v in metrics.items():
                mlflow.log_metric(k, v, step=server_round)
            mlflow.log_param("round", server_round)
        # Save snapshot
        save_global(parameters)
        return loss, metrics
    return evaluate

if __name__ == "__main__":
    rounds = int(os.getenv("NUM_ROUNDS", "3"))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=3,
        min_evaluate_clients=0,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda r: {"local_epochs": 3},
    )
    print(f"Starting Flower server for {rounds} rounds...")
    fl.server.start_server(server_address="[::]:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=rounds))
