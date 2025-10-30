import os, json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.model_selection import ParameterGrid

from utils import ensure_dir, select_X_y, save_json, DEFAULT_TARGET

SPLIT_DIR   = r"PoC\Model-Development-Pipeline\artifacts\splits"
MODEL_DIR   = r"PoC\Model-Development-Pipeline\artifacts\models_federated"
REPORT_DIR  = r"PoC\Model-Development-Pipeline\artifacts\reports_federated"
META_DIR    = r"PoC\Model-Development-Pipeline\artifacts\meta_features"

RANDOM_STATE = 42
TARGET_COL   = DEFAULT_TARGET  # "icu_mortality_label"
HOSPITALS    = ["hospital_a", "hospital_b", "hospital_c"]

# ---------- helpers ----------

def load_split(name: str, split: str) -> pd.DataFrame:
    path = os.path.join(SPLIT_DIR, f"{name}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def _class_weights(y: np.ndarray) -> dict:
    """Compute inverse frequency class weights as {0: w0, 1: w1}."""
    vals, cnts = np.unique(y, return_counts=True)
    total = cnts.sum()
    weights = {int(v): total / (len(vals) * c) for v, c in zip(vals, cnts)}
    # ensure both present
    if 0 not in weights: weights[0] = 1.0
    if 1 not in weights: weights[1] = 1.0
    return weights

def _sample_weight_from_class_weights(y: np.ndarray, w: dict) -> np.ndarray:
    return np.array([w[int(t)] for t in y], dtype=float)

def _safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return np.nan

def _eval_binary(y_true, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "roc_auc": _safe_auc(y_true, p),
        "pr_auc": float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": float(f1_score(y_true, yhat)) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, yhat)),
        "n": int(len(y_true)),
    }

# ---------- local model training ----------

def train_local_model(df_train: pd.DataFrame, df_val: pd.DataFrame):
    # select_X_y should: drop target from X, coerce numerics where possible (your utils does this)
    Xtr, ytr = select_X_y(df_train, target=TARGET_COL)
    Xva, yva = select_X_y(df_val, target=TARGET_COL)

    # align val columns to training set (order + any missing)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)

    # class balancing via sample weights for HGB
    cw = _class_weights(ytr)
    sw = _sample_weight_from_class_weights(ytr, cw)

    # a compact *robust* grid for tiny data
    grid = ParameterGrid({
        "max_depth": [None, 6, 10],
        "learning_rate": [0.05, 0.1],
        "max_iter": [200, 400],
        "l2_regularization": [0.0, 1.0],
        "min_samples_leaf": [10, 20],           # stabilize on small sets
        "max_leaf_nodes": [31, 63],             # reasonable trees
    })

    best_model, best_score, best_params = None, -np.inf, None

    for params in grid:
        clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        clf.fit(Xtr, ytr, sample_weight=sw)

        # predict on validation
        p = clf.predict_proba(Xva)[:, 1]
        yhat = (p >= 0.5).astype(int)

        auc = _safe_auc(yva, p)
        metric_score = auc if not np.isnan(auc) else accuracy_score(yva, yhat)
        if metric_score > best_score:
            best_score, best_model, best_params = metric_score, clf, params

    # summarize validation metrics for the best model
    p = best_model.predict_proba(Xva)[:, 1]
    metrics = {
        **_eval_binary(yva, p),
        "params": best_params,
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_classes_val": int(len(np.unique(yva))),
    }

    # bundle for later aligned scoring
    bundle = {"model": best_model, "feature_names": Xtr.columns.tolist()}
    return bundle, metrics

def score_with_locals(models_bundle, df: pd.DataFrame) -> pd.DataFrame:
    """Return stacked meta-features with columns p_A, p_B, p_C (in that order)."""
    X, _ = select_X_y(df, target=TARGET_COL)
    out = {}
    for key, bundle in models_bundle.items():
        feats = bundle["feature_names"]
        X_aligned = X.reindex(columns=feats, fill_value=0.0)
        out[key] = bundle["model"].predict_proba(X_aligned)[:, 1]
    return pd.DataFrame({"p_A": out["A"], "p_B": out["B"], "p_C": out["C"]}, index=df.index)

# ---------- main ----------

def main():
    ensure_dir(MODEL_DIR); ensure_dir(REPORT_DIR); ensure_dir(META_DIR)

    # 1) Train local models per hospital on (train -> tune on validation)
    local_models = {}
    local_metrics = {}

    for hosp, tag in zip(HOSPITALS, ["A", "B", "C"]):
        print(f"\n=== Training local model: {hosp} ===")
        df_tr = load_split(hosp, "train")
        df_va = load_split(hosp, "validation")   # keep *validation* as requested

        # guard: empty validation after any upstream filtering (shouldn’t happen with repaired splits)
        if len(df_va) == 0:
            raise ValueError(f"{hosp}: validation split is empty.")

        bundle, metrics = train_local_model(df_tr, df_va)
        local_models[tag] = bundle
        local_metrics[hosp] = metrics

        # save local model
        path = os.path.join(MODEL_DIR, f"{hosp}_HGB.joblib")
        joblib.dump(bundle, path)
        print(f"Saved local model -> {path}")
        print("Val metrics:", json.dumps(metrics, indent=2))

    save_json(local_metrics, os.path.join(REPORT_DIR, "local_val_metrics.json"))

    # 2) Build stacked features for global splits
    print("\n=== Building stacked features for global ===")
    df_g_tr = load_split("global", "train")
    df_g_va = load_split("global", "validation")
    df_g_te = load_split("global", "test")

    y_tr = df_g_tr[TARGET_COL].astype(int).values
    y_va = df_g_va[TARGET_COL].astype(int).values
    y_te = df_g_te[TARGET_COL].astype(int).values

    S_tr = score_with_locals({"A": local_models["A"], "B": local_models["B"], "C": local_models["C"]}, df_g_tr)
    S_va = score_with_locals({"A": local_models["A"], "B": local_models["B"], "C": local_models["C"]}, df_g_va)
    S_te = score_with_locals({"A": local_models["A"], "B": local_models["B"], "C": local_models["C"]}, df_g_te)

    # Save meta-features (debug/inspection)
    ensure_dir(META_DIR)
    S_tr.assign(**{TARGET_COL: y_tr}).to_csv(os.path.join(META_DIR, "global_train_meta.csv"), index=False)
    S_va.assign(**{TARGET_COL: y_va}).to_csv(os.path.join(META_DIR, "global_val_meta.csv"), index=False)
    S_te.assign(**{TARGET_COL: y_te}).to_csv(os.path.join(META_DIR, "global_test_meta.csv"), index=False)

    # 3) Train global meta-learner (LogReg) with class_weight='balanced'
    print("\n=== Training global meta-learner (stacking) ===")
    meta = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    meta.fit(S_tr, y_tr)

    # Evaluate meta on validation + test
    p_va = meta.predict_proba(S_va)[:, 1]
    meta_val_metrics = _eval_binary(y_va, p_va)
    print("Global meta (val) metrics:", json.dumps(meta_val_metrics, indent=2))
    save_json(meta_val_metrics, os.path.join(REPORT_DIR, "global_meta_val_metrics.json"))

    meta_path = os.path.join(MODEL_DIR, "global_meta_lr.joblib")
    joblib.dump({"model": meta, "features": ["p_A", "p_B", "p_C"]}, meta_path)
    print(f"Saved meta-learner -> {meta_path}")

    p_te = meta.predict_proba(S_te)[:, 1]
    meta_test_metrics = _eval_binary(y_te, p_te)
    save_json(meta_test_metrics, os.path.join(REPORT_DIR, "global_meta_test_metrics.json"))
    print("Global meta (test) metrics:", json.dumps(meta_test_metrics, indent=2))

    print("\n✓ Federated stacking training complete.")

if __name__ == "__main__":
    main()
