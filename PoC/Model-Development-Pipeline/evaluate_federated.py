# PoC/Model-Developement-Pipeline/evaluate_federated.py
import os, json
import joblib
import pandas as pd
from utils import select_X_y, DEFAULT_TARGET

SPLIT_DIR   = r"PoC\Model-Developement-Pipeline\artifacts\splits"
MODEL_DIR   = r"PoC\Model-Developement-Pipeline\artifacts\models_federated"
REPORT_DIR  = r"PoC\Model-Developement-Pipeline\artifacts\reports_federated"
TARGET_COL  = DEFAULT_TARGET

def load_split(name: str, split: str) -> pd.DataFrame:
    path = os.path.join(SPLIT_DIR, f"{name}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def score(df: pd.DataFrame, bundle) -> pd.Series:
    X, _ = select_X_y(df, target=TARGET_COL)
    X = X.reindex(columns=bundle["feature_names"], fill_value=0.0)
    return bundle["model"].predict_proba(X)[:, 1]

def main():
    # load global test
    df_test = load_split("global", "test")
    y = df_test[TARGET_COL].astype(int)

    # load local models
    A = joblib.load(os.path.join(MODEL_DIR, "hospital_a_HGB.joblib"))
    B = joblib.load(os.path.join(MODEL_DIR, "hospital_b_HGB.joblib"))
    C = joblib.load(os.path.join(MODEL_DIR, "hospital_c_HGB.joblib"))

    # stacked features
    S = pd.DataFrame({
        "p_A": score(df_test, A),
        "p_B": score(df_test, B),
        "p_C": score(df_test, C),
    })

    # load meta
    META = joblib.load(os.path.join(MODEL_DIR, "global_meta_lr.joblib"))["model"]
    p = META.predict_proba(S)[:, 1]

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
    yhat = (p >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, yhat)),
        "accuracy": float(accuracy_score(y, yhat)),
        "n_test": int(len(y))
    }

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "global_meta_test_metrics_rerun.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("\n✓ Federated evaluation complete.")

if __name__ == "__main__":
    main()
