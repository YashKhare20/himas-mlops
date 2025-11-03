"""
Validates and writes per-hospital split files from federated CSVs.
Optionally enforces minimum class presence in validation/test.
"""

import os
import pandas as pd
from config import (
    FEDERATED_DIR as FEDERATED,
    SPLITS_OUT_DIR as OUT_DIR,
    SPLIT_COL,
    VALID_SPLITS,
    TARGET_COL as CFG_TARGET_COL,
    ENFORCE_MIN_PER_CLASS,
    MIN_PER_CLASS,
)
from utils import ensure_dir, DEFAULT_TARGET


LABEL_COL = CFG_TARGET_COL or DEFAULT_TARGET
REQUIRED_COLS = {"subject_id", SPLIT_COL, LABEL_COL}


HAS_PARQUET = False
_PARQUET_ENGINE = None
try:
    import pyarrow  # noqa
    HAS_PARQUET = True
    _PARQUET_ENGINE = "pyarrow"
except Exception:
    try:
        import fastparquet  # noqa
        HAS_PARQUET = True
        _PARQUET_ENGINE = "fastparquet"
    except Exception:
        HAS_PARQUET = False
        _PARQUET_ENGINE = None


def load_hospital(letter: str) -> pd.DataFrame:
    """
    Load a hospital CSV and validate required columns and split labels.

    Args:
        letter: Hospital letter suffix, for example 'a' for hospital_a.

    Returns:
        DataFrame with normalized split labels and integer label column.
    """
    path = os.path.join(FEDERATED, f"hospital_{letter}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    df = df.copy()
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    bad = set(df[SPLIT_COL].unique()) - VALID_SPLITS
    if bad:
        raise ValueError(f"{path} has unexpected split labels: {bad}")

    if not pd.api.types.is_integer_dtype(df[LABEL_COL]):
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    return df


def _stable_subjects(df: pd.DataFrame) -> list:
    """
    Return a stable list of subject IDs.

    Args:
        df: Input DataFrame containing subject_id.

    Returns:
        Sorted list of unique subject IDs.
    """
    try:
        return sorted(df["subject_id"].unique().tolist())
    except Exception:
        return sorted(df["subject_id"].astype(str).unique().tolist())


def _need_for_split(df: pd.DataFrame, split_name: str) -> dict:
    """
    Compute how many subjects are needed per class to reach the minimum quota.

    Args:
        df: Hospital DataFrame.
        split_name: Target split name.

    Returns:
        Dict mapping class value to required count to reach minimum.
    """
    cur = df[df[SPLIT_COL] == split_name]
    counts = cur[LABEL_COL].value_counts()
    need = {}
    for cls in (0, 1):
        have = int(counts.get(cls, 0))
        if have < MIN_PER_CLASS:
            need[cls] = MIN_PER_CLASS - have
    return need


def _move_subjects(df: pd.DataFrame, split_name: str, cls: int, deficit: int):
    """
    Move subjects of a given class from train to a target split.

    Args:
        df: Hospital DataFrame.
        split_name: Target split to move into.
        cls: Class label to move.
        deficit: Number of subjects to move.

    Returns:
        Tuple of (updated DataFrame, log messages list).
    """
    logs = []
    if deficit <= 0:
        return df, logs
    train = df[df[SPLIT_COL] == "train"]
    if train.empty:
        logs.append(f"[{split_name}] cannot repair: no rows in train.")
        return df, logs
    cand_subjects = _stable_subjects(train[train[LABEL_COL] == cls][["subject_id"]])
    moved = []
    for sid in cand_subjects:
        if deficit <= 0:
            break
        mask = (df["subject_id"] == sid) & (df[SPLIT_COL] == "train")
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        df.loc[mask, SPLIT_COL] = split_name
        moved.append((sid, n_rows))
        deficit -= 1
    if moved:
        moved_txt = ", ".join([f"{sid}({n})" for sid, n in moved])
        logs.append(f"[{split_name}] moved {len(moved)} subject(s) of class {cls} from train -> {split_name}: {moved_txt}")
    else:
        logs.append(f"[{split_name}] could not find train subjects of class {cls} to move.")
    return df, logs


def maybe_repair(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Optionally enforce minimum per-class presence in validation and test.

    Args:
        df: Hospital DataFrame.
        name: Hospital name used in logs.

    Returns:
        Potentially modified DataFrame with logs printed.
    """
    if not ENFORCE_MIN_PER_CLASS:
        return df
    all_logs = []
    for split_name in ["validation", "test"]:
        need = _need_for_split(df, split_name)
        for cls, deficit in need.items():
            df, logs = _move_subjects(df, split_name, cls, deficit)
            all_logs.extend(logs)
        need_after = _need_for_split(df, split_name)
        for cls, deficit in need_after.items():
            if deficit > 0:
                all_logs.append(f"[{split_name}] still missing class {cls} by {deficit} after repair (insufficient train).")
    for line in all_logs:
        print(f"{name} {line}")
    return df


def _row(df: pd.DataFrame, split_name: str):
    """
    Summarize counts for a single split.

    Args:
        df: Hospital DataFrame.
        split_name: Split name.

    Returns:
        Tuple of (n_total, n_class0, n_class1).
    """
    g = df.groupby([SPLIT_COL, LABEL_COL]).size().unstack(fill_value=0)
    if split_name not in g.index:
        return 0, 0, 0
    r = g.loc[split_name]
    return int(r.sum()), int(r.get(0, 0)), int(r.get(1, 0))


def print_split_summary(df: pd.DataFrame, name: str) -> None:
    """
    Print a brief per-split summary for a hospital.

    Args:
        df: Hospital DataFrame.
        name: Hospital name.

    Returns:
        None.
    """
    for split_name in ["train", "validation", "test"]:
        n, z, o = _row(df, split_name)
        print(f"{name} {split_name:<11s} -> (n={n}) | {LABEL_COL}: 0={z}, 1={o}")


def _safe_to_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Write to Parquet if a supported engine is available.

    Args:
        df: DataFrame to write.
        path: Output file path.

    Returns:
        None.
    """
    if not HAS_PARQUET:
        print(f"[skip] parquet write disabled (no engine). Would write: {path}")
        return
    df.to_parquet(path, index=False, engine=_PARQUET_ENGINE)


def save_by_split(df: pd.DataFrame, out_prefix: str) -> None:
    """
    Save per-split CSV and Parquet files for a hospital.

    Args:
        df: Hospital DataFrame.
        out_prefix: Filename prefix such as 'hospital_a'.

    Returns:
        None.
    """
    ensure_dir(OUT_DIR)
    for split_name in ["train", "validation", "test"]:
        part = df[df[SPLIT_COL] == split_name]
        if len(part) == 0:
            continue
        csv_path = os.path.join(OUT_DIR, f"{out_prefix}_{split_name}.csv")
        pq_path = os.path.join(OUT_DIR, f"{out_prefix}_{split_name}.parquet")
        part.to_csv(csv_path, index=False)
        _safe_to_parquet(part, pq_path)


def main() -> None:
    """
    Load hospital CSVs, optionally repair class presence, summarize, and save per-split files.
    """
    hA = load_hospital("a")
    hB = load_hospital("b")
    hC = load_hospital("c")

    hA = maybe_repair(hA, "hospital_a")
    hB = maybe_repair(hB, "hospital_b")
    hC = maybe_repair(hC, "hospital_c")

    print_split_summary(hA, "hospital_a")
    print_split_summary(hB, "hospital_b")
    print_split_summary(hC, "hospital_c")

    save_by_split(hA, "hospital_a")
    save_by_split(hB, "hospital_b")
    save_by_split(hC, "hospital_c")


if __name__ == "__main__":
    main()
