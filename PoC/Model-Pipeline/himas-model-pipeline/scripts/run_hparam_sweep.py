"""
run_hparam_sweep.py
===================

Centralized hyperparameter sweep for the ICU mortality MLP.

- Loads data from BigQuery for all hospitals (train/val splits)
- Uses the same DataPreprocessor + load_model() as the federated pipeline
- Runs multiple training jobs with different hyperparameters
- Logs everything to MLflow (one run per hyperparameter set)
- Computes a simple "sensitivity" ranking of which hyperparameters
  move validation AUC the most (within this sweep)
- Finally trains/evaluates the **current baseline configuration**
  (shared hyperparameters from pyproject/JSON) and shows where it ranks.

This script is deliberately independent from Flower/Server/Client code:
it does not start federated training and does not modify any global
state used by those components.
"""

import argparse
import itertools
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.utils.class_weight import compute_class_weight

from himas_model_pipeline.task import (
    DataPreprocessor,
    get_config_value,
    get_shared_hyperparameters,
    load_model,
    set_seed,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hparam_sweep")


# ---------------------------------------------------------------------
# Data loading (centralized, NOT federated)
# ---------------------------------------------------------------------
def load_centralized_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/validation data from ALL hospitals and concatenate.

    This is ONLY for hyperparameter exploration. The federated pipeline
    still uses hospital-specific preprocessors and local training.
    """
    project_id = get_config_value("tool.himas.data.project-id")
    dataset_id = get_config_value("tool.himas.data.dataset-id")
    hospital_names = get_config_value(
        "tool.himas.data.hospital-names",
        ["hospital_a", "hospital_b", "hospital_c"],
    )
    train_split = get_config_value("tool.himas.data.train-split", "train")
    val_split = get_config_value("tool.himas.data.validation-split", "validation")

    client = bigquery.Client(project=project_id)

    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []

    logger.info("Loading centralized TRAIN/VAL data from BigQuery")
    for hosp in hospital_names:
        table = f"{project_id}.{dataset_id}.{hosp}_data"

        q_train = f"SELECT * FROM `{table}` WHERE data_split = '{train_split}'"
        q_val = f"SELECT * FROM `{table}` WHERE data_split = '{val_split}'"

        df_train = client.query(q_train).to_dataframe()
        df_val = client.query(q_val).to_dataframe()

        df_train["__hospital__"] = hosp
        df_val["__hospital__"] = hosp

        logger.info(
            "  %s: %d train, %d val",
            hosp,
            len(df_train),
            len(df_val),
        )

        train_frames.append(df_train)
        val_frames.append(df_val)

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True)

    logger.info(
        "Centralized data: %d train samples, %d val samples",
        len(train_df),
        len(val_df),
    )
    return train_df, val_df


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Fit DataPreprocessor on centralized TRAIN, then transform VAL.
    Returns X_train, y_train, X_val, y_val, input_dim.
    """
    pre = DataPreprocessor()
    X_train, y_train = pre.fit_transform(train_df)
    X_val, y_val = pre.transform(val_df)

    input_dim = pre.feature_dim
    assert input_dim is not None

    logger.info("Preprocessing complete:")
    logger.info("  X_train: %s, X_val: %s, input_dim=%d", X_train.shape, X_val.shape, input_dim)
    return X_train, y_train, X_val, y_val, input_dim


# ---------------------------------------------------------------------
# Hyperparameter definitions
# ---------------------------------------------------------------------
@dataclass
class HparamConfig:
    num_layers: int
    architecture: str
    first_layer_units: int
    activation: str
    dropout_rate: float
    l2_strength: float
    learning_rate: float
    optimizer: str

    label: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("label", None)
        return d


def build_search_space() -> List[HparamConfig]:
    """
    Define a small, sane grid of hyperparameters to explore.

    You can expand this if you want a deeper search.
    """
    num_layers = [3, 4]
    architectures = ["decreasing", "uniform"]
    first_units = [128, 256]
    dropout_rates = [0.2, 0.4]

    # Keep these fixed; they're already reasonable defaults.
    activation = "relu"
    l2_strength = 1e-3
    learning_rate = 1e-3
    optimizer = "adam"

    space: List[HparamConfig] = []
    for i, (nl, arch, units, dr) in enumerate(
        itertools.product(num_layers, architectures, first_units, dropout_rates), start=1
    ):
        label = f"sweep_{i:02d}_L{nl}_{arch}_U{units}_DR{dr}"
        space.append(
            HparamConfig(
                num_layers=nl,
                architecture=arch,
                first_layer_units=units,
                activation=activation,
                dropout_rate=dr,
                l2_strength=l2_strength,
                learning_rate=learning_rate,
                optimizer=optimizer,
                label=label,
            )
        )
    return space


# ---------------------------------------------------------------------
# Training / evaluation for a single config
# ---------------------------------------------------------------------
def train_and_eval_single_config(
    hp: HparamConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    base_seed: int,
    run_group: str,
    is_baseline: bool = False,
) -> Dict[str, Any]:
    """
    Train and evaluate one hyperparameter configuration.
    Logs everything to MLflow and returns a result dict.
    """
    import keras

    # Derive a deterministic seed for this run
    seed = base_seed + (hash(hp.label) % 10_000)
    set_seed(seed)

    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "himas-federated-hparam")
    mlflow.set_experiment(exp_name)

    run_name = hp.label if hp.label else "current_baseline"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags(
            {
                "role": "centralized-tuning",
                "group": run_group,
                "is_baseline": str(is_baseline),
            }
        )
        mlflow.log_params(hp.as_dict())
        mlflow.log_param("seed", seed)
        mlflow.log_param("input_dim", input_dim)

        # Build model
        model = load_model(input_dim=input_dim, hyperparameters=hp.as_dict(), seed=seed)

        # Training setup
        max_epochs = 30
        batch_size = 256

        classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        logger.info(
            "Training %s (baseline=%s) for up to %d epochs, batch_size=%d",
            run_name,
            is_baseline,
            max_epochs,
            batch_size,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            class_weight=cw,
            callbacks=callbacks,
            shuffle=True,
            verbose=1,
        )

        # Final metrics from best epoch (last in history after ES)
        hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        last_idx = -1
        val_auc = hist["val_auc"][last_idx]
        val_loss = hist["val_loss"][last_idx]
        val_acc = hist["val_accuracy"][last_idx]
        val_prec = hist["val_precision"][last_idx]
        val_rec = hist["val_recall"][last_idx]
        val_f1 = (
            2 * val_prec * val_rec / (val_prec + val_rec)
            if (val_prec + val_rec) > 0
            else 0.0
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
                "n_train": int(len(X_train)),
                "n_val": int(len(X_val)),
                "epochs_run": int(len(hist["loss"])),
            }
        )

        # Save history as an artifact for later inspection
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("hparam_results")
        out_dir.mkdir(exist_ok=True)
        hist_path = out_dir / f"{run_name}_history_{ts}.json"
        with open(hist_path, "w") as f:
            json.dump(hist, f, indent=2)
        mlflow.log_artifact(str(hist_path), artifact_path="training_history")

        result = {
            "label": run_name,
            "is_baseline": is_baseline,
            "val_auc": val_auc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "epochs_run": len(hist["loss"]),
            "run_id": run_id,
            "hyperparameters": hp.as_dict(),
        }

    return result


# ---------------------------------------------------------------------
# Simple sensitivity summary
# ---------------------------------------------------------------------
def summarize_sensitivity(results: List[Dict[str, Any]]) -> None:
    """
    Very simple, local "sensitivity" analysis:
    - For each hyperparameter, look at the spread of val_auc across its values.
    - Larger spread = more impact on performance within this sweep.
    """
    if not results:
        logger.warning("No results to analyze for sensitivity.")
        return

    df = pd.DataFrame(
        [
            {
                **{"label": r["label"], "val_auc": r["val_auc"], "is_baseline": r["is_baseline"]},
                **r["hyperparameters"],
            }
            for r in results
        ]
    )

    logger.info("=" * 70)
    logger.info("Hyperparameter Sensitivity (local sweep, based on val_auc)")
    logger.info("=" * 70)

    hp_cols = [
        "num_layers",
        "architecture",
        "first_layer_units",
        "dropout_rate",
        "l2_strength",
        "learning_rate",
        "optimizer",
    ]

    rows = []
    for col in hp_cols:
        if col not in df.columns:
            continue

        grouped = df.groupby(col)["val_auc"].agg(["mean", "min", "max"])
        if len(grouped) <= 1:
            continue

        effect_range = grouped["max"].max() - grouped["min"].min()
        rows.append(
            {
                "hyperparameter": col,
                "effect_range": float(effect_range),
                "n_values": int(len(grouped)),
            }
        )

    if not rows:
        logger.info("No varying hyperparameters found in this sweep.")
        return

    eff_df = (
        pd.DataFrame(rows)
        .sort_values("effect_range", ascending=False)
        .reset_index(drop=True)
    )

    for _, r in eff_df.iterrows():
        logger.info(
            "  %-18s  effect_range=%.4f  (across %d values)",
            r["hyperparameter"],
            r["effect_range"],
            r["n_values"],
        )


def print_ranking(results: List[Dict[str, Any]]) -> None:
    """
    Print a ranking of all runs by val_auc and highlight the current baseline.
    """
    sorted_results = sorted(results, key=lambda r: r["val_auc"], reverse=True)

    logger.info("=" * 70)
    logger.info("Validation AUC Ranking (higher is better)")
    logger.info("=" * 70)
    for rank, r in enumerate(sorted_results, start=1):
        tag = "  <-- CURRENT CONFIG" if r["is_baseline"] else ""
        logger.info(
            "%2d. %-30s  val_auc=%.4f  val_f1=%.4f%s",
            rank,
            r["label"],
            r["val_auc"],
            r["val_f1"],
            tag,
        )


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centralized hyperparameter sweep for ICU mortality MLP"
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=8,
        help="Maximum number of sweep configs to evaluate (default: 8)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed for sweep (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_seed = args.base_seed

    # 1) Load and preprocess centralized data
    train_df, val_df = load_centralized_data()
    X_train, y_train, X_val, y_val, input_dim = preprocess_data(train_df, val_df)

    # 2) Build hyperparameter search space
    search_space = build_search_space()
    if args.max_runs < len(search_space):
        logger.info(
            "Restricting sweep to first %d configurations out of %d.",
            args.max_runs,
            len(search_space),
        )
        search_space = search_space[: args.max_runs]

    sweep_results: List[Dict[str, Any]] = []
    run_group = datetime.now().strftime("hparam_sweep_%Y%m%d_%H%M%S")

    # 3) Run sweep
    for hp in search_space:
        result = train_and_eval_single_config(
            hp=hp,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim,
            base_seed=base_seed,
            run_group=run_group,
            is_baseline=False,
        )
        sweep_results.append(result)

    # 4) Run CURRENT baseline configuration
    logger.info("=" * 70)
    logger.info("Running CURRENT baseline hyperparameters for comparison")
    logger.info("=" * 70)

    current_hp_dict = get_shared_hyperparameters(context=None)
    if current_hp_dict is None:
        logger.info(
            "No shared hyperparameters JSON found, using defaults from pyproject.toml"
        )
        # Use load_model defaults to infer HP; but for logging we still need a dict
        current_hp_dict = {
            "num_layers": get_config_value("tool.himas.model.num-layers", 4),
            "architecture": get_config_value(
                "tool.himas.model.architecture", "decreasing"
            ),
            "first_layer_units": get_config_value(
                "tool.himas.model.first-layer-units", 256
            ),
            "activation": get_config_value("tool.himas.model.activation", "relu"),
            "dropout_rate": get_config_value("tool.himas.model.dropout-rate", 0.3),
            "l2_strength": get_config_value("tool.himas.model.l2-strength", 0.001),
            "learning_rate": get_config_value(
                "tool.himas.model.learning-rate", 0.001
            ),
            "optimizer": get_config_value("tool.himas.model.optimizer", "adam"),
        }

    baseline_hp = HparamConfig(
        label="current_baseline",
        **current_hp_dict,
    )

    baseline_result = train_and_eval_single_config(
        hp=baseline_hp,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        base_seed=base_seed,
        run_group=run_group,
        is_baseline=True,
    )
    sweep_results.append(baseline_result)

    # 5) Local sensitivity summary + ranking
    summarize_sensitivity(sweep_results)
    print_ranking(sweep_results)

    logger.info("Hyperparameter sweep completed.")


if __name__ == "__main__":
    main()
