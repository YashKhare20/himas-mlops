# generate_samples.py
# Builds ICU-only features from the MIMIC-IV demo and splits them into
# three hospital CSVs + a global test set for the federated POC.

import argparse, json, os
from pathlib import Path
import pandas as pd
import numpy as np


# ---------- Config (only from poc/data/config.json unless overridden) ----------

def load_config() -> dict:
    """Load poc/data/config.json if present, else {}."""
    cfg_path = Path(__file__).resolve().parent / "config.json"  # poc/data/config.json
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def cfg_get(cfg: dict, path: str, default=None):
    """Dot-path getter for nested dicts."""
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------- Helpers to find data ----------

def find_curated(start: Path) -> Path | None:
    """Look for *_curated/icu_features_gold.parquet upward and downward."""
    candidates = list(start.rglob("icu_features_gold.parquet"))
    candidates = [c for c in candidates if c.parent.name == "_curated"]
    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        return candidates[0]
    return None


def _is_mimic_base(p: Path) -> bool:
    """A valid base has hosp/ and icu/ with the key files present."""
    try:
        if not p.is_dir():
            return False
        hosp = p / "hosp"
        icu  = p / "icu"
        if not (hosp.exists() and icu.exists()):
            return False
        has_pats = (hosp / "patients.csv.gz").exists() or (hosp / "patients.csv").exists()
        has_adm  = (hosp / "admissions.csv.gz").exists() or (hosp / "admissions.csv").exists()
        has_lab  = (hosp / "labevents.csv.gz").exists()  or (hosp / "labevents.csv").exists()
        has_icu  = (icu  / "icustays.csv.gz").exists()   or (icu  / "icustays.csv").exists()
        return has_pats and has_adm and has_lab and has_icu
    except Exception:
        return False


def find_mimic_base(start: Path) -> Path | None:
    """
    Find a folder that DIRECTLY contains `hosp/` and `icu/`.
    Handles:
      - mimic-iv-demo-2.2
      - mimic-iv-clinical-database-demo-2.2
      - nested duplicate folders (zip-inside-zip layout)
      - sibling projects (Pipeline vs Pipleine)
    """
    start = start.resolve()

    # Likely relative guesses
    guesses = [
        "mimic-iv-demo-2.2",
        "mimic-iv-clinical-database-demo-2.2",
        "mimic-iv-clinical-database-demo-2.2/mimic-iv-clinical-database-demo-2.2",
        r"mimic-iv-clinical-database-demo-2.2\mimic-iv-clinical-database-demo-2.2",
        "mimic-iv-clinical-database-demo-2.2/mimic-iv-demo-2.2",
        r"mimic-iv-clinical-database-demo-2.2\mimic-iv-demo-2.2",
    ]
    for g in guesses:
        p = (start / g).resolve()
        if _is_mimic_base(p):
            return p

    # Walk ancestors; inspect their children & grandchildren
    to_check = []
    for parent in [start] + list(start.parents):
        to_check.append(parent)
        try:
            for child in parent.iterdir():
                if child.is_dir():
                    to_check.append(child)
                    for grand in child.iterdir():
                        if grand.is_dir():
                            to_check.append(grand)
        except PermissionError:
            pass

    seen = set()
    for root in to_check:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if _is_mimic_base(root):
            return root
        try:
            for hosp_dir in root.rglob("hosp"):
                cand = hosp_dir.parent
                if _is_mimic_base(cand):
                    return cand.resolve()
        except Exception:
            pass

    return None


# ---------- IO helpers ----------

def read_first(base: Path, relpaths: list[str]) -> pd.DataFrame | None:
    """Return the first CSV(.gz) found under base from relpaths, else None."""
    for rel in relpaths:
        p = base / rel
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    return None


# ---------- Minimal rebuild from raw ----------

def minimal_rebuild_from_raw(mimic_base: Path) -> pd.DataFrame:
    """Rebuild a small ICU features table from raw demo CSVs."""
    patients   = read_first(mimic_base, ["hosp/patients.csv.gz", "hosp/patients.csv"])
    admissions = read_first(mimic_base, ["hosp/admissions.csv.gz", "hosp/admissions.csv"])
    icustays   = read_first(mimic_base, ["icu/icustays.csv.gz", "icu/icustays.csv"])
    labevents  = read_first(mimic_base, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])

    # parse datetimes
    for df, cols in (
        (admissions, ["admittime", "dischtime", "deathtime"]),
        (icustays,   ["intime", "outtime"]),
        (labevents,  ["charttime"]),
    ):
        if df is not None:
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")

    # slim
    pat_s = patients[["subject_id", "gender", "anchor_age"]].copy() if patients is not None else None
    adm_s = admissions[["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"]].copy() if admissions is not None else None
    icu_s = icustays[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"]].copy() if icustays is not None else None

    if pat_s is None or adm_s is None or icu_s is None:
        raise SystemExit("Missing raw demo tables; cannot rebuild ICU features.")

    icu_cohort = (
        icu_s
        .merge(adm_s, on=["subject_id", "hadm_id"], how="left")
        .merge(pat_s, on="subject_id", how="left")
    )
    icu_cohort["icu_los_hrs"] = (icu_cohort["outtime"] - icu_cohort["intime"]).dt.total_seconds() / 3600.0

    # ICU-window labs (mean per itemid)
    if labevents is not None and not labevents.empty:
        labs = labevents[["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]].copy()
        labs = labs.dropna(subset=["valuenum"])  # avoid empty-mean warnings
        labs = labs.merge(icu_s[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id", how="inner")
        mask = (labs["charttime"] >= labs["intime"]) & (labs["charttime"] <= labs["outtime"])
        labs = labs.loc[mask].copy()
        agg  = labs.groupby(["stay_id", "itemid"])["valuenum"].agg(["mean"]).reset_index()
        wide = agg.pivot(index="stay_id", columns="itemid", values="mean")
        wide.columns = [f"lab_mean_{c}" for c in wide.columns]
        icu_features = icu_cohort.merge(wide, left_on="stay_id", right_index=True, how="left")
    else:
        icu_features = icu_cohort.copy()

    return icu_features


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    # Defaults: out_dir is this file's folder (poc/data). mimic_base may come from config.
    ap.add_argument("--out_dir", type=str, default="", help="Output data folder (default: this script's folder)")
    ap.add_argument("--mimic_base", type=str, default="", help="Folder that DIRECTLY contains `hosp/` and `icu/`")
    ap.add_argument("--los_threshold_hours", type=float, default=None, help="Prolonged ICU LOS threshold (default 48.0)")
    ap.add_argument("--test_frac",  type=float, default=0.20, help="Global test fraction (unused; deterministic split)")
    args = ap.parse_args()

    # Load config (poc/data/config.json)
    cfg = load_config()

    # Resolve sources with precedence: CLI > ENV > config.json > default/auto
    mimic_base = (
        args.mimic_base.strip()
        or os.environ.get("MIMIC_BASE", "").strip()
        or (cfg_get(cfg, "data.mimic_base", "").strip())
    )

    out_dir_cfg = cfg_get(cfg, "data.out_dir", "").strip()
    out_dir = (
        Path(args.out_dir).as_posix() if args.out_dir
        else os.environ.get("POC_OUT_DIR", out_dir_cfg or "")
    )
    if out_dir:
        out_dir = Path(out_dir).resolve()
    else:
        # default to this script's folder (poc/data)
        out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    los_thr = (
        args.los_threshold_hours
        if args.los_threshold_hours is not None
        else float(cfg_get(cfg, "build.los_threshold_hours", 48.0))
    )

    start_here = Path(".").resolve()

    # 1) Try curated parquet first
    curated = None
    if mimic_base:
        mb = Path(mimic_base).resolve()
        curated = mb / "_curated" / "icu_features_gold.parquet"
        if not curated.exists():
            curated = None
    if curated is None:
        curated = find_curated(start_here)

    if curated and curated.exists():
        print(f"[OK] Using curated ICU features: {curated}")
        df = pd.read_parquet(curated)
    else:
        # 2) Rebuild from raw CSVs: prefer explicit path, else auto-find
        chosen_base = Path(mimic_base).resolve() if mimic_base else find_mimic_base(start_here)
        if not (chosen_base and _is_mimic_base(chosen_base)):
            print("[HINT] We tried to find a folder with BOTH `hosp/` and `icu/` and key CSV(.gz) files.")
            print("[HINT] Example (based on your tree):")
            print("       C:\\College\\MLOPS\\Data_Pipeline_POC\\mimic-iv-clinical-database-demo-2.2\\mimic-iv-clinical-database-demo-2.2")
            raise SystemExit(
                "No curated parquet found and demo base not auto-detected.\n"
                "Pass --mimic_base <PATH> or set it in poc/data/config.json "
                "pointing to the folder that DIRECTLY contains `hosp/` and `icu/`."
            )
        print(f"[INFO] Rebuilding minimal ICU features from raw demo CSVs at: {chosen_base}")
        df = minimal_rebuild_from_raw(chosen_base)

    # 3) Label: prolonged ICU stay
    if "icu_los_hrs" not in df.columns:
        raise SystemExit("icu_los_hrs missing; cannot make label.")
    df["target_prolonged_icu"] = (df["icu_los_hrs"] >= float(los_thr)).astype(int)

    # 4) Select features
    lab_cols = [c for c in df.columns if c.startswith("lab_mean_")]
    base_cols = ["anchor_age"] if "anchor_age" in df.columns else []
    cat_cols  = [c for c in ["gender", "first_careunit"] if c in df.columns]

    use_cols = ["subject_id", "hadm_id", "stay_id", "icu_los_hrs", "target_prolonged_icu"] + base_cols + cat_cols + lab_cols
    df = df[[c for c in use_cols if c in df.columns]].copy()

    # 5) One-hot encode categoricals
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 6) Impute & scale numeric features (minmax 0-1)
    label_col = "target_prolonged_icu"
    id_cols   = ["subject_id", "hadm_id", "stay_id", "icu_los_hrs", label_col]
    feat_cols = [c for c in df.columns if c not in id_cols]

    # Ensure numeric features only
    # 6a) bool -> uint8
    bool_cols = df[feat_cols].select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("uint8")

    # 6b) object -> numeric (coerce)
    obj_cols = df[feat_cols].select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 6c) drop any remaining non-numeric features
    num_cols = df[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    dropped = [c for c in feat_cols if c not in num_cols]
    if dropped:
        print("[WARN] Dropping non-numeric feature(s):", dropped)
        feat_cols = num_cols

    # median impute numeric
    for c in feat_cols:
        df[c] = df[c].fillna(df[c].median())

    mins = df[feat_cols].min()
    maxs = df[feat_cols].max()
    rngs = (maxs - mins).replace(0, 1.0)  # safe range; no division by zero
    df[feat_cols] = (df[feat_cols] - mins) / rngs

    # 7) Deterministic splits by subject_id (global test ≈20%, then 3 hospitals)
    # De-fragment once, then assign BOTH columns at once to avoid warnings
    df = df.copy()
    sid = df["subject_id"].astype(int).to_numpy()
    df = df.assign(_mod3=(sid % 3), _mod10=(sid % 10))

    global_test = df[df["_mod10"].isin([0, 1])].copy()
    train_pool  = df.loc[~df.index.isin(global_test.index)].copy()

    hosp1 = train_pool[train_pool["_mod3"] == 0].copy()
    hosp2 = train_pool[train_pool["_mod3"] == 1].copy()
    hosp3 = train_pool[train_pool["_mod3"] == 2].copy()

    cols_out = ["subject_id", "hadm_id", "stay_id", label_col] + feat_cols
    out_dir.mkdir(parents=True, exist_ok=True)
    hosp1[cols_out].to_csv(out_dir / "hosp1.csv", index=False)
    hosp2[cols_out].to_csv(out_dir / "hosp2.csv", index=False)
    hosp3[cols_out].to_csv(out_dir / "hosp3.csv", index=False)
    global_test[cols_out].to_csv(out_dir / "global_test.csv", index=False)

    scaler_info = {"mins": mins.to_dict(), "maxs": maxs.to_dict(), "rngs": rngs.to_dict()}
    (out_dir / "scaler_info.json").write_text(json.dumps(scaler_info, indent=2))

    meta = {"label": label_col, "features": feat_cols, "classes": [0, 1], "n_features": len(feat_cols)}
    (out_dir / "feature_list.json").write_text(json.dumps(meta, indent=2))

    print("[DONE] Wrote under:", out_dir.resolve())
    for f in ["hosp1.csv", "hosp2.csv", "hosp3.csv", "global_test.csv", "feature_list.json", "scaler_info.json"]:
        print(" -", (out_dir / f).resolve())


if __name__ == "__main__":
    main()
