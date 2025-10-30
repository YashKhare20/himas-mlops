
import numpy as np, pandas as pd, json, os
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import os, mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "federated-learning"))

DATA_DIR = Path(os.getenv("DATA_DIR", "poc/data"))
ART_DIR  = Path(os.getenv("ART_DIR", "poc/artifacts"))

meta = json.loads((DATA_DIR/"feature_list.json").read_text())
FEATURES = meta["features"]; LABEL = meta["label"]; CLASSES = np.array(meta["classes"], dtype=int)

bundle = np.load(ART_DIR/"global_model.npz", allow_pickle=True)
coef = bundle["coef"]; intercept = bundle["intercept"]

df = pd.read_csv(DATA_DIR/"global_test.csv")
X = df[FEATURES].values.astype(float)
y = df[LABEL].values.astype(int)

clf = SGDClassifier(loss="log_loss", max_iter=1, random_state=0)
clf.partial_fit(X[:min(32,len(X))], y[:min(32,len(y))], classes=CLASSES)
clf.coef_ = coef.reshape(clf.coef_.shape)
clf.intercept_ = intercept
clf.classes_ = CLASSES

yhat = clf.predict(X)
acc  = accuracy_score(y, yhat)
print("ACC:", acc)
try:
    proba = clf.predict_proba(X)[:,1]
    print("AUC:", roc_auc_score(y, proba))
    print("LogLoss:", log_loss(y, proba))
except Exception: pass
