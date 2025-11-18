# scripts/model_sensitivity_analysis.py

"""
HIMAS Model Sensitivity Analysis
================================

Post-hoc analysis tools for the federated ICU mortality model.

Features
--------
1. Feature Importance (SHAP):
   - Uses the latest saved federated model
   - Reuses the same leakage-safe preprocessing used in evaluation
   - Computes SHAP values on a held-out sample
   - Produces:
       * Bar chart of global feature importance (mean |SHAP|)
       * Optional SHAP summary (beeswarm) plot
       * JSON export of per-feature importance scores

2. Hyperparameter Sensitivity (MLflow-based):
   - Reads past training runs from the MLflow experiment
   - Analyzes how variations in logged hyperparameters
     relate to validation AUC
   - Produces:
       * Bar chart ranking hyperparameters by effect size
       * JSON summary of parameter-metric relationships

Usage
-----
From the project root (where pyproject.toml lives):

  # Run full sensitivity suite (feature importance + hyperparams)
  python -m scripts.model_sensitivity_analysis

  # Only feature importance (SHAP)
  python -m scripts.model_sensitivity_analysis --skip-hparam

  # Only hyperparameter sensitivity
  python -m scripts.model_sensitivity_analysis --skip-shap

  # Use a smaller sample for faster SHAP computations
  python -m scripts.model_sensitivity_analysis --sample-size 500 --background-size 100

The outputs are written under:
  evaluation_results/
    figures/<model_name>/
      feature_importance_shap_bar.png
      feature_importance_shap_summary.png
    results/
      shap_feature_importance_<timestamp>.json
      hparam_sensitivity_<timestamp>.json
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import keras
import mlflow
import numpy as np
import pandas as pd
import shap
from google.cloud import bigquery
from matplotlib import pyplot as plt

# Reuse config + preprocessing from evaluation script
from scripts.evaluate_model import (  # type: ignore
    PROJECT_ID,
    DATASET_ID,
    HOSPITALS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    DataPreprocessor,
    get_config_value,
    get_latest_model_path,
)

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Helpers: data loading for SHAP
# -------------------------------------------------------------------------
def load_training_data_for_shap(project_id: str) -> pd.DataFrame:
    """Load combined TRAIN data across all hospitals (for fitting preprocessor)."""
    client = bigquery.Client(project=project_id)
    train_split = get_config_value("tool.himas.data.train-split", "train")

    frames = []
    for hosp in HOSPITALS:
        query = (
            f"SELECT * FROM `{project_id}.{DATASET_ID}.{hosp}_data` "
            f"WHERE data_split = '{train_split}'"
        )
        df = client.query(query).to_dataframe()
        logger.info("Loaded %s train samples from %s", len(df), hosp)
        frames.append(df)

    all_train = pd.concat(frames, ignore_index=True)
    logger.info("Combined train size for SHAP preprocessor: %s", len(all_train))
    return all_train


def load_test_data_for_shap(project_id: str) -> pd.DataFrame:
    """Load combined TEST data across all hospitals (for SHAP explanations)."""
    client = bigquery.Client(project=project_id)
    test_split = get_config_value("tool.himas.data.test-split", "test")

    frames = []
    for hosp in HOSPITALS:
        query = (
            f"SELECT * FROM `{project_id}.{DATASET_ID}.{hosp}_data` "
            f"WHERE data_split = '{test_split}'"
        )
        df = client.query(query).to_dataframe()
        df["__hospital__"] = hosp
        logger.info("Loaded %s test samples from %s", len(df), hosp)
        frames.append(df)

    all_test = pd.concat(frames, ignore_index=True)
    logger.info("Combined test size for SHAP explanations: %s", len(all_test))
    return all_test


# -------------------------------------------------------------------------
# Feature Importance via SHAP
# -------------------------------------------------------------------------
class ShapFeatureImportanceAnalyzer:
    """Compute global feature importance for the federated model using SHAP."""

    def __init__(
        self,
        model_path: Path,
        project_id: str,
        sample_size: int = 1000,
        background_size: int = 200,
    ) -> None:
        self.model_path = model_path
        self.project_id = project_id
        self.sample_size = sample_size
        self.background_size = background_size

        self.model: keras.Model | None = None
        self.preprocessor: DataPreprocessor | None = None

        eval_dir = Path(
            get_config_value("tool.himas.paths.evaluation-dir", "evaluation_results")
        )
        self.output_dir = eval_dir
        self.figures_dir = eval_dir / "figures" / self.model_path.stem
        self.results_dir = eval_dir / "results"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Feature order matches DataPreprocessor (numerical first, then categorical)
        self.feature_names: List[str] = list(NUMERICAL_FEATURES) + list(
            CATEGORICAL_FEATURES
        )

    # ---------------------- public API ---------------------------------
    def run(self) -> None:
        """End-to-end SHAP analysis."""
        logger.info("Starting SHAP feature importance analysis")
        self._load_model_and_fit_preprocessor()
        X_bg, X_sample = self._prepare_background_and_sample()
        shap_values = self._compute_shap_values(X_bg, X_sample)
        self._summarize_and_save_results(shap_values, X_sample)

    # ---------------------- internal helpers ---------------------------
    def _load_model_and_fit_preprocessor(self) -> None:
        """Load trained Keras model and fit preprocessor on TRAIN data."""
        logger.info("=" * 70)
        logger.info("SHAP: Loading model and fitting preprocessor")
        logger.info("=" * 70)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        self.model = keras.models.load_model(str(self.model_path))
        logger.info("Loaded model from %s", self.model_path)
        logger.info("  Parameters: %s", f"{self.model.count_params():,}")

        # Fit preprocessor on combined TRAIN across all hospitals
        df_train = load_training_data_for_shap(self.project_id)
        self.preprocessor = DataPreprocessor()
        self.preprocessor.fit(df_train)
        logger.info("Preprocessor fitted on TRAIN data")

    def _prepare_background_and_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare background and sample matrices for SHAP."""
        assert self.preprocessor is not None

        df_test = load_test_data_for_shap(self.project_id)

        # Sample rows for SHAP analysis (for speed)
        if len(df_test) > self.sample_size:
            df_sample = df_test.sample(self.sample_size, random_state=42)
        else:
            df_sample = df_test

        # Background sample used by SHAP (smaller)
        if len(df_test) > self.background_size:
            df_bg = df_test.sample(self.background_size, random_state=123)
        else:
            df_bg = df_test

        X_bg, _ = self.preprocessor.transform(df_bg)
        X_sample, _ = self.preprocessor.transform(df_sample)

        logger.info("SHAP background shape: %s", X_bg.shape)
        logger.info("SHAP sample shape: %s", X_sample.shape)
        return X_bg, X_sample

    def _compute_shap_values(
        self, X_bg: np.ndarray, X_sample: np.ndarray
    ) -> np.ndarray:
        """Compute SHAP values for the positive class."""
        assert self.model is not None

        logger.info("Initializing SHAP DeepExplainer (this can take a bit)...")
        explainer = shap.DeepExplainer(self.model, X_bg)

        logger.info("Computing SHAP values for sample...")
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values is a list of length 1
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        logger.info("SHAP values shape: %s", shap_values.shape)
        return shap_values

    def _summarize_and_save_results(
        self, shap_values: np.ndarray, X_sample: np.ndarray
    ) -> None:
        """Create plots + JSON summaries for global feature importance."""
        logger.info("Summarizing SHAP results")

        # --------------------------------------------------------------
        # 1) Normalize SHAP values to 2D: (n_samples, n_features)
        # --------------------------------------------------------------
        shap_arr = np.array(shap_values)
        logger.info("Original SHAP values shape: %s", shap_arr.shape)

        if shap_arr.ndim == 3 and shap_arr.shape[-1] == 1:
            shap_arr = shap_arr.squeeze(-1)  # (n_samples, n_features)
            logger.info("Squeezed SHAP values shape: %s", shap_arr.shape)
        elif shap_arr.ndim == 2:
            logger.info("SHAP values already 2D")
        else:
            raise ValueError(
                f"Unexpected SHAP values shape {shap_arr.shape}; "
                "expected 2D or 3D with last dim = 1."
            )

        n_samples, n_features = shap_arr.shape
        logger.info(
            "Using SHAP values with shape (n_samples=%d, n_features=%d)",
            n_samples,
            n_features,
        )

        # --------------------------------------------------------------
        # 2) Align feature names with actual SHAP feature dimension
        # --------------------------------------------------------------
        feature_names = list(self.feature_names[:n_features])
        if len(feature_names) != n_features:
            logger.warning(
                "Number of feature names (%d) does not match SHAP feature "
                "dimension (%d); truncating to minimum.",
                len(feature_names),
                n_features,
            )
            m = min(len(feature_names), n_features)
            feature_names = feature_names[:m]
            shap_arr = shap_arr[:, :m]
            X_sample = X_sample[:, :m]
            n_features = m

        # --------------------------------------------------------------
        # 3) Aggregate global importance: mean |SHAP| per feature (1D)
        # --------------------------------------------------------------
        mean_abs_shap = np.mean(np.abs(shap_arr), axis=0).ravel()
        logger.info("mean_abs_shap shape: %s", mean_abs_shap.shape)

        importance_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": mean_abs_shap,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        # --------------------------------------------------------------
        # 4) Bar plot of global importance
        # --------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        plt.barh(
            importance_df["feature"],
            importance_df["mean_abs_shap"],
        )
        plt.gca().invert_yaxis()
        plt.xlabel("Mean |SHAP value|")
        plt.title("Global Feature Importance (SHAP)\nICU Mortality Model")
        plt.tight_layout()
        bar_path = self.figures_dir / "feature_importance_shap_bar.png"
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("SHAP bar plot saved to %s", bar_path)

        # --------------------------------------------------------------
        # 5) Full SHAP summary (beeswarm) plot
        # --------------------------------------------------------------
        shap.summary_plot(
            shap_arr,
            X_sample,
            feature_names=feature_names,
            show=False,
        )
        summary_path = self.figures_dir / "feature_importance_shap_summary.png"
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("SHAP summary plot saved to %s", summary_path)

        # --------------------------------------------------------------
        # 6) JSON export
        # --------------------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.results_dir / f"shap_feature_importance_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "model_path": str(self.model_path),
                    "timestamp": datetime.now().isoformat(),
                    "features": importance_df.to_dict(orient="records"),
                },
                f,
                indent=2,
            )
        logger.info("SHAP importance JSON saved to %s", json_path)


# -------------------------------------------------------------------------
# Hyperparameter Sensitivity via MLflow
# -------------------------------------------------------------------------
def analyze_hyperparameter_sensitivity(
    experiment_name: str = "himas-federated",
    metric_key: str = "val_auc",
    output_dir: Path | None = None,
) -> None:
    """
    Analyze how logged hyperparameters influence a target metric (e.g. val_auc).

    This uses existing MLflow runs. It does *not* retrain models.
    It is most useful after you’ve run multiple experiments with different
    hyperparameters (e.g. different shared hyperparameter JSONs, batch sizes,
    local epochs, learning rates, etc.).
    """
    logger.info("=" * 70)
    logger.info("Hyperparameter Sensitivity via MLflow")
    logger.info("  Experiment: %s", experiment_name)
    logger.info("  Target metric: %s", metric_key)
    logger.info("=" * 70)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment(experiment_name)
        df = mlflow.search_runs(experiment_names=[experiment_name])
    except Exception as e:
        logger.warning("Could not query MLflow runs: %s", e)
        return

    if df.empty:
        logger.warning("No runs found in MLflow for experiment '%s'", experiment_name)
        return

    metric_col = f"metrics.{metric_key}"
    if metric_col not in df.columns:
        logger.warning(
            "Metric '%s' not present in MLflow runs; available metrics: %s",
            metric_key,
            [c for c in df.columns if c.startswith("metrics.")],
        )
        return

    # Identify hyperparameter columns (params.*) that have more than 1 distinct value
    param_cols = [c for c in df.columns if c.startswith("params.")]
    varying_params = [
        c for c in param_cols if df[c].dropna().nunique() > 1
    ]

    if not varying_params:
        logger.warning(
            "No varying hyperparameters found across runs – "
            "cannot compute sensitivity."
        )
        return

    logger.info("Found %d varying hyperparameters", len(varying_params))

    # Compute simple effect size: range of mean metric across param values
    rows = []
    for col in varying_params:
        grouped = df.groupby(col)[metric_col].mean().dropna()
        if len(grouped) < 2:
            continue

        effect = grouped.max() - grouped.min()
        rows.append(
            {
                "param": col.replace("params.", ""),
                "min_metric": float(grouped.min()),
                "max_metric": float(grouped.max()),
                "effect_range": float(effect),
                "n_values": int(len(grouped)),
            }
        )

    if not rows:
        logger.warning("No usable param–metric relationships found.")
        return

    sensitivity_df = pd.DataFrame(rows).sort_values(
        "effect_range", ascending=False
    )

    logger.info("Top hyperparameters by effect_range on %s:", metric_key)
    for _, r in sensitivity_df.head(10).iterrows():
        logger.info(
            "  %-25s  effect_range=%.4f  (min=%.4f, max=%.4f, n_values=%d)",
            r["param"],
            r["effect_range"],
            r["min_metric"],
            r["max_metric"],
            r["n_values"],
        )

    if output_dir is None:
        output_dir = Path(
            get_config_value("tool.himas.paths.evaluation-dir", "evaluation_results")
        )
    output_dir.mkdir(exist_ok=True)

    # Bar plot
    plt.figure(figsize=(10, 6))
    plt.barh(
        sensitivity_df["param"],
        sensitivity_df["effect_range"],
    )
    plt.gca().invert_yaxis()
    plt.xlabel(f"Range of mean {metric_key} across values")
    plt.title(f"Hyperparameter Sensitivity on {metric_key}")
    plt.tight_layout()
    fig_path = output_dir / "hyperparameter_sensitivity.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Hyperparameter sensitivity plot saved to %s", fig_path)

    # JSON summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"hparam_sensitivity_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "metric": metric_key,
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": sensitivity_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    logger.info("Hyperparameter sensitivity JSON saved to %s", json_path)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HIMAS model sensitivity analysis (SHAP + hyperparameters)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of test samples to explain with SHAP (default: 1000)",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=200,
        help="Background sample size for SHAP (default: 200)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="himas-federated-eval",
        help="MLflow experiment name for hyperparameter analysis",
    )
    parser.add_argument(
        "--metric-key",
        type=str,
        default="val_auc",
        help="Metric name used in MLflow for hyperparameter sensitivity (e.g. 'val_auc')",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP feature importance analysis",
    )
    parser.add_argument(
        "--skip-hparam",
        action="store_true",
        help="Skip hyperparameter sensitivity analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine latest model produced by federated training
    model_path = get_latest_model_path()
    logger.info("Using model for sensitivity analysis: %s", model_path)

    if not args.skip_shap:
        shap_analyzer = ShapFeatureImportanceAnalyzer(
            model_path=model_path,
            project_id=PROJECT_ID,
            sample_size=args.sample_size,
            background_size=args.background_size,
        )
        shap_analyzer.run()
    else:
        logger.info("Skipping SHAP feature importance (per CLI flag).")

    if not args.skip_hparam:
        analyze_hyperparameter_sensitivity(
            experiment_name=args.experiment_name,
            metric_key=args.metric_key,
        )
    else:
        logger.info("Skipping hyperparameter sensitivity (per CLI flag).")


if __name__ == "__main__":
    main()
