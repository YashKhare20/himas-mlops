import os
import json
import hashlib
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

# -------- Defaults --------
RANDOM_STATE = 42
# unified target name used across scripts
DEFAULT_TARGET = "icu_mortality_label"

# -------- IO helpers --------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# -------- Splitting helpers (fallback only) --------
def hash_bucket(subject_id: str, modulo: int = 100) -> int:
    """Deterministic hash bucket (stringify subject_id first)."""
    h = hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()
    return int(h, 16) % modulo

def assign_split_by_subject(subject_ids: pd.Series) -> pd.Series:
    """
    Deterministic split per patient:
      - 70% train, 20% val, 10% test based on hash bucket.
    Used only if no split is provided by data.
    """
    buckets = subject_ids.astype(str).apply(hash_bucket, modulo=10)
    cond_train = buckets < 7
    cond_val   = (buckets >= 7) & (buckets < 9)
    out = pd.Series(np.where(cond_train, "train", np.where(cond_val, "val", "test")),
                    index=subject_ids.index)
    return out

# -------- Feature/label prep --------
def select_X_y(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    drop_cols: List[str] = ("subject_id", "stay_id", "hospital", "split", "data_split", "assigned_hospital")
) -> Tuple[pd.DataFrame, pd.Series]:
    """Keep only numeric features; drop ids/metadata and the target."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")
    y = df[target].astype(int)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target], errors="ignore")
    # Keep only numeric; coerce where possible
    X = X.apply(
    lambda s: pd.to_numeric(s, errors="coerce") if s.dtype == "object" else s)
    X = X.fillna(0)
    X = X.select_dtypes(include=["number"]).copy()
    # Clean
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y

# -------- Metrics / thresholding --------
def threshold_search(y_true: np.ndarray, p: np.ndarray,
                     metric: str = "f1",
                     grid: Optional[np.ndarray] = None) -> dict:
    """
    Simple threshold sweep to pick a good operating point.
    Returns dict(threshold, f1, accuracy, precision, recall).
    """
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)  # 0.1,0.15,...,0.9

    best = {"threshold": 0.5, "f1": -1, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    if len(np.unique(y_true)) < 2:
        # No search possible
        yhat = (p >= 0.5).astype(int)
        best.update({
            "threshold": 0.5,
            "f1": float(f1_score(y_true, yhat, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, yhat)),
            "precision": float(precision_score(y_true, yhat, zero_division=0)),
            "recall": float(recall_score(y_true, yhat, zero_division=0)),
        })
        return best

    for t in grid:
        yhat = (p >= t).astype(int)
        f1 = f1_score(y_true, yhat, zero_division=0)
        acc = accuracy_score(y_true, yhat)
        pre = precision_score(y_true, yhat, zero_division=0)
        rec = recall_score(y_true, yhat, zero_division=0)
        score = f1 if metric == "f1" else acc
        if score > best[metric]:
            best = {"threshold": float(t), "f1": float(f1), "accuracy": float(acc),
                    "precision": float(pre), "recall": float(rec)}
    return best
