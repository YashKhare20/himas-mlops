import os
import pandas as pd
from utils import ensure_dir, DEFAULT_TARGET

DATA_ROOT  = r"PoC\Data-Pipeline\data"
FEDERATED  = os.path.join(DATA_ROOT, "federated_demo")
OUT_DIR    = r"PoC\Model-Development-Pipeline\artifacts\splits"

LABEL_COL      = DEFAULT_TARGET          # "icu_mortality_label"
VALID_SPLITS   = {"train", "validation", "test"}
MIN_PER_CLASS  = 1                       # ensure ≥1 of each class in validation/test

REQUIRED_COLS  = {"subject_id", "data_split", LABEL_COL}

# ---------- loading ----------

def load_hospital(letter: str) -> pd.DataFrame:
    path = os.path.join(FEDERATED, f"hospital_{letter}_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    df = pd.read_csv(path)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    df = df.copy()
    # keep exact labels but normalize case/whitespace; do NOT rename "validation"
    df["data_split"] = df["data_split"].astype(str).str.strip().str.lower()
    bad = set(df["data_split"].unique()) - VALID_SPLITS
    if bad:
        raise ValueError(f"{path} has unexpected split labels: {bad}")

    # normalize label to integer 0/1 when possible
    if not pd.api.types.is_integer_dtype(df[LABEL_COL]):
        df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)

    return df

# ---------- helpers ----------

def _stable_subjects(df: pd.DataFrame) -> list:
    """Deterministic subject ordering."""
    try:
        return sorted(df["subject_id"].unique().tolist())
    except Exception:
        return sorted(df["subject_id"].astype(str).unique().tolist())

def _need_for_split(df: pd.DataFrame, split_name: str) -> dict:
    """Return deficits per class to ensure MIN_PER_CLASS for 0 and 1."""
    cur = df[df["data_split"] == split_name]
    counts = cur[LABEL_COL].value_counts()
    need = {}
    for cls in (0, 1):
        have = int(counts.get(cls, 0))
        if have < MIN_PER_CLASS:
            need[cls] = MIN_PER_CLASS - have
    return need

def _move_subjects(df: pd.DataFrame, split_name: str, cls: int, deficit: int) -> tuple[pd.DataFrame, list]:
    """
    Move entire subjects from train -> split_name that contain at least
    one row of class `cls`. Return (df, move_logs).
    """
    logs = []
    if deficit <= 0:
        return df, logs

    train = df[df["data_split"] == "train"]
    if train.empty:
        logs.append(f"[{split_name}] cannot repair: no rows in train.")
        return df, logs

    cand_subjects = _stable_subjects(train[train[LABEL_COL] == cls][["subject_id"]])
    moved = []
    for sid in cand_subjects:
        if deficit <= 0:
            break
        mask = (df["subject_id"] == sid) & (df["data_split"] == "train")
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        df.loc[mask, "data_split"] = split_name
        moved.append((sid, n_rows))
        deficit -= 1  # subject-level granularity

    if moved:
        moved_txt = ", ".join([f"{sid}({n})" for sid, n in moved])
        logs.append(f"[{split_name}] moved {len(moved)} subject(s) of class {cls} from train -> {split_name}: {moved_txt}")
    else:
        logs.append(f"[{split_name}] could not find train subjects of class {cls} to move.")

    return df, logs

def repair_hospital(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure each of validation and test has ≥ MIN_PER_CLASS for both 0 and 1
    by moving whole subjects from train. Keeps split names exactly as provided.
    """
    all_logs = []
    for split_name in ["validation", "test"]:
        need = _need_for_split(df, split_name)
        # try to fix missing classes by pulling from train
        for cls, deficit in need.items():
            df, logs = _move_subjects(df, split_name, cls, deficit)
            all_logs.extend(logs)

        # re-check; if still missing, print a note (we won't move across val<->test)
        need_after = _need_for_split(df, split_name)
        for cls, deficit in need_after.items():
            if deficit > 0:
                all_logs.append(f"[{split_name}] still missing class {cls} by {deficit} after repair (insufficient train).")

    for line in all_logs:
        print(f"{name:>10s} {line}")
    return df

def _row(df: pd.DataFrame, split_name: str):
    g = df.groupby(["data_split", LABEL_COL]).size().unstack(fill_value=0)
    if split_name not in g.index:
        return 0, 0, 0
    r = g.loc[split_name]
    return int(r.sum()), int(r.get(0, 0)), int(r.get(1, 0))

def print_split_summary(df: pd.DataFrame, name: str) -> None:
    for split_name in ["train", "validation", "test"]:
        n, z, o = _row(df, split_name)
        print(f"{name:>10s} {split_name:<11s} -> (n={n}) | {LABEL_COL}: 0={z}, 1={o}")

def save_by_split(df: pd.DataFrame, out_prefix: str) -> None:
    ensure_dir(OUT_DIR)
    for split_name in ["train", "validation", "test"]:
        part = df[df["data_split"] == split_name]
        if len(part) == 0:
            continue
        part.to_csv(os.path.join(OUT_DIR, f"{out_prefix}_{split_name}.csv"), index=False)
        part.to_parquet(os.path.join(OUT_DIR, f"{out_prefix}_{split_name}.parquet"), index=False)

def make_global(hA: pd.DataFrame, hB: pd.DataFrame, hC: pd.DataFrame) -> pd.DataFrame:
    # concatenate after per-hospital repairs; keep existing split labels
    return pd.concat([hA, hB, hC], axis=0, ignore_index=True)

# ---------- main ----------

def main():
    # load
    hA = load_hospital("a")
    hB = load_hospital("b")
    hC = load_hospital("c")

    # repair: ensure validation/test each have ≥1 of both labels
    hA = repair_hospital(hA, "hospital_a")
    hB = repair_hospital(hB, "hospital_b")
    hC = repair_hospital(hC, "hospital_c")

    # summaries (after repair)
    print_split_summary(hA, "hospital_a")
    print_split_summary(hB, "hospital_b")
    print_split_summary(hC, "hospital_c")

    # save per-hospital
    save_by_split(hA, "hospital_a")
    save_by_split(hB, "hospital_b")
    save_by_split(hC, "hospital_c")

    # global (concat repaired)
    g = make_global(hA, hB, hC)
    print_split_summary(g, "global")
    save_by_split(g, "global")

if __name__ == "__main__":
    main()
