"""
HIMAS Federated Model Evaluation Script
Evaluates the trained federated learning model on test data from all hospitals.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

import keras
from google.cloud import bigquery
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# --- MLflow (minimal additions) ---
import os
import mlflow

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
_MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT_NAME", "himas-federated-eval")
mlflow.set_tracking_uri(_MLFLOW_URI)
mlflow.set_experiment(_MLFLOW_EXP)

# Configuration
PROJECT_ID = "erudite-carving-472018-r5"
# DATASET_ID = "federated"
DATASET_ID = "federated_demo"
MODEL_PATH = "models/himas_federated_mortality_model.keras"

HOSPITALS = ["hospital_a", "hospital_b", "hospital_c"]

# Feature columns (must match training configuration)
NUMERICAL_FEATURES = [
    'age_at_admission', 'los_icu_hours', 'los_icu_days',
    'los_hospital_days', 'los_hospital_hours', 'n_distinct_icu_units',
    'is_mixed_icu', 'n_icu_transfers', 'n_total_transfers',
    'ed_admission_flag', 'emergency_admission_flag', 'hours_admit_to_icu',
    'early_icu_score', 'weekend_admission', 'night_admission'
]

CATEGORICAL_FEATURES = [
    'icu_type', 'first_careunit', 'admission_type',
    'admission_location', 'insurance', 'gender', 'race', 'marital_status'
]

TARGET = 'icu_mortality_label'


class ModelEvaluator:
    """Comprehensive evaluation of federated ICU mortality prediction model."""

    def __init__(self, model_path: str, project_id: str):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the saved Keras model
            project_id: Google Cloud project ID
        """
        self.model_path = Path(model_path)
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.model = None
        self.results = {}

        # Create output directories
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

    def load_model(self):
        """Load the trained federated model."""
        print(f"\nLoading model from {self.model_path}...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model = keras.models.load_model(str(self.model_path))
        print("Model loaded successfully")
        print(f"  Model input shape: {self.model.input_shape}")
        print(f"  Model output shape: {self.model.output_shape}")

    def load_test_data(self, hospital: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load test data from BigQuery for a specific hospital.

        Args:
            hospital: Hospital name (hospital_a, hospital_b, or hospital_c)

        Returns:
            Tuple of (dataframe, features, labels)
        """
        table_name = f"{hospital}_data"

        query = f"""
        SELECT *
        FROM `{self.project_id}.{DATASET_ID}.{table_name}`
        WHERE data_split = 'test'
        """

        print(f"\nLoading test data for {hospital}...")
        df = self.client.query(query).to_dataframe()
        print(f"  Test samples: {len(df)}")
        print(f"  Mortality rate: {df[TARGET].mean():.2%}")

        # Preprocess features
        X = self.preprocess_features(df)
        y = df[TARGET].values

        return df, X, y

    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features to match training preprocessing.

        Args:
            df: Raw dataframe from BigQuery

        Returns:
            Preprocessed feature matrix
        """
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        # Extract feature columns
        X = df.drop(columns=[TARGET, 'stay_id', 'subject_id', 'hadm_id',
                             'icu_intime', 'icu_outtime', 'deathtime', 'death_date',
                             'assigned_hospital', 'data_split'])

        # Handle numerical features - Convert to float64 first to handle BigQuery Int64 dtype
        X_numerical = X[NUMERICAL_FEATURES].copy()

        # Convert all numerical columns to float64 to avoid Int64 fillna issues
        for col in NUMERICAL_FEATURES:
            if col in X_numerical.columns:
                X_numerical[col] = pd.to_numeric(X_numerical[col], errors='coerce').astype('float64')

        # Now compute median and fillna with float values
        X_numerical = X_numerical.fillna(X_numerical.median())

        # Standardize
        scaler = StandardScaler()
        X_numerical_scaled = scaler.fit_transform(X_numerical)

        # Handle categorical features
        X_categorical_encoded = []
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                le = LabelEncoder()
                col_data = X[col].fillna('Unknown').astype(str)
                encoded = le.fit_transform(col_data)
                X_categorical_encoded.append(encoded.reshape(-1, 1))

        # Combine features
        if X_categorical_encoded:
            X_categorical_array = np.hstack(X_categorical_encoded)
            X_processed = np.hstack([X_numerical_scaled, X_categorical_array])
        else:
            X_processed = X_numerical_scaled

        return X_processed

    def evaluate_hospital(self, hospital: str) -> Dict:
        """
        Evaluate model performance on a single hospital's test data.

        Args:
            hospital: Hospital name

        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data
        df, X_test, y_test = self.load_test_data(hospital)

        # Make predictions
        print(f"\nEvaluating {hospital}...")
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        metrics = {
            'hospital': hospital,
            'n_samples': len(y_test),
            'n_positive': int(y_test.sum()),
            'n_negative': int(len(y_test) - y_test.sum()),
            'prevalence': float(y_test.mean()),
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'average_precision': float(average_precision_score(y_test, y_pred_proba)),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

        # Clinical metrics
        metrics['specificity'] = float(
            tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        # Negative Predictive Value
        metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        metrics['ppv'] = metrics['precision']  # Positive Predictive Value

        # Store for visualization
        self.results[hospital] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba.flatten(),
            'df': df
        }

        return metrics

    def evaluate_all_hospitals(self) -> List[Dict]:
        """
        Evaluate model on all hospitals and compute aggregated metrics.

        Returns:
            List of metric dictionaries for each hospital
        """
        hospital_metrics = []

        for hospital in HOSPITALS:
            m = self.evaluate_hospital(hospital)
            hospital_metrics.append(m)
            # Log each hospital's metrics to MLflow with a step index
            mlflow.log_metrics(
                {k: v for k, v in m.items() if isinstance(v, (int, float))},
                step=len(hospital_metrics),
            )

        # Aggregate predictions across all hospitals
        all_y_test = np.concatenate([self.results[h]['y_test'] for h in HOSPITALS])
        all_y_pred = np.concatenate([self.results[h]['y_pred'] for h in HOSPITALS])
        all_y_pred_proba = np.concatenate([self.results[h]['y_pred_proba'] for h in HOSPITALS])
        all_df = pd.concat([self.results[h]['df'] for h in HOSPITALS], ignore_index=True)

        aggregated_metrics = {
            'hospital': 'AGGREGATED',
            'n_samples': len(all_y_test),
            'n_positive': int(all_y_test.sum()),
            'n_negative': int(len(all_y_test) - all_y_test.sum()),
            'prevalence': float(all_y_test.mean()),
            'accuracy': float(accuracy_score(all_y_test, all_y_pred)),
            'precision': float(precision_score(all_y_test, all_y_pred, zero_division=0)),
            'recall': float(recall_score(all_y_test, all_y_pred, zero_division=0)),
            'f1_score': float(f1_score(all_y_test, all_y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(all_y_test, all_y_pred_proba)),
            'average_precision': float(average_precision_score(all_y_test, all_y_pred_proba)),
        }

        cm = confusion_matrix(all_y_test, all_y_pred)
        tn, fp, fn, tp = cm.ravel()

        aggregated_metrics['confusion_matrix'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        aggregated_metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        aggregated_metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        aggregated_metrics['ppv'] = aggregated_metrics['precision']

        # store aggregated results for plotting/ reporting
        self.results['AGGREGATED'] = {
            'metrics': aggregated_metrics,
            'y_test': all_y_test,
            'y_pred': all_y_pred,
            'y_pred_proba': all_y_pred_proba,
            'df': all_df,
        }

        hospital_metrics.append(aggregated_metrics)
        # Log aggregated metrics
        mlflow.log_metrics(
            {k: v for k, v in aggregated_metrics.items() if isinstance(v, (int, float))},
            step=0,
        )
        return hospital_metrics

    # --- plotting helpers unchanged (omitted here for brevity; keep your originals) ---
    # [All plotting methods remain as in your current file,
    #  they save PNGs into evaluation_results/figures/]

    # (Keep all the existing plotting functions exactly as you have them.)
    # ... (your existing _plot_roc_curves, _plot_precision_recall_curves, _plot_confusion_matrices,
    #      _plot_metrics_comparison, _plot_prediction_distribution)

    # (The rest of the class remains unchanged; only MLflow logging added in evaluate_all_hospitals
    #  and the run wrapper in main().)

    def generate_visualizations(self):
        """Generate comprehensive visualization plots."""
        print("\n" + "="*60)
        print("Generating Visualizations")
        print("="*60)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # 1. ROC Curves for all hospitals
        self._plot_roc_curves()

        # 2. Precision-Recall Curves
        self._plot_precision_recall_curves()

        # 3. Confusion Matrices
        self._plot_confusion_matrices()

        # 4. Metrics Comparison
        self._plot_metrics_comparison()

        # 5. Prediction Distribution
        self._plot_prediction_distribution()

        print("All visualizations saved")

    # [Keep all five plot functions exactly as in your file]

    def _plot_roc_curves(self):
        """Plot ROC curves for each hospital and aggregated data."""
        fig, ax = plt.subplots()

        for hospital in HOSPITALS:
            y_true = self.results[hospital]['y_test']
            y_score = self.results[hospital]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, label=f"{hospital} (AUC={auc_val:.3f})")

        if 'AGGREGATED' in self.results:
            y_true = self.results['AGGREGATED']['y_test']
            y_score = self.results['AGGREGATED']['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)
            ax.plot(
                fpr,
                tpr,
                linestyle="--",
                linewidth=2,
                label=f"All hospitals (AUC={auc_val:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves by Hospital")
        ax.legend(loc="lower right")
        fig.tight_layout()

        out_path = self.output_dir / "figures" / "roc_curves.png"
        fig.savefig(out_path)
        plt.close(fig)

    def _plot_precision_recall_curves(self):
        """Plot precision-recall curves for each hospital and aggregated data."""
        fig, ax = plt.subplots()

        for hospital in HOSPITALS:
            y_true = self.results[hospital]['y_test']
            y_score = self.results[hospital]['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, label=f"{hospital} (AP={ap:.3f})")

        if 'AGGREGATED' in self.results:
            y_true = self.results['AGGREGATED']['y_test']
            y_score = self.results['AGGREGATED']['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(
                recall,
                precision,
                linestyle="--",
                linewidth=2,
                label=f"All hospitals (AP={ap:.3f})",
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves by Hospital")
        ax.legend(loc="lower left")
        fig.tight_layout()

        out_path = self.output_dir / "figures" / "precision_recall_curves.png"
        fig.savefig(out_path)
        plt.close(fig)

    def _plot_confusion_matrices(self):
        """Plot confusion matrices for each hospital and aggregated data."""
        hospitals_to_plot = HOSPITALS + (['AGGREGATED'] if 'AGGREGATED' in self.results else [])
        n = len(hospitals_to_plot)
        cols = 2
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, hospital in zip(axes, hospitals_to_plot):
            res = self.results[hospital]
            y_true = res['y_test']
            y_pred = res['y_pred']
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
            ax.set_title(hospital.replace("_", " ").title())

        for ax in axes[len(hospitals_to_plot):]:
            ax.axis("off")

        fig.suptitle("Confusion Matrices", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = self.output_dir / "figures" / "confusion_matrices.png"
        fig.savefig(out_path)
        plt.close(fig)

    def _plot_metrics_comparison(self):
        """Compare key metrics across hospitals and aggregated."""
        records = []
        for hospital in HOSPITALS:
            m = self.results[hospital]['metrics']
            records.append(
                {
                    "hospital": hospital,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1_score": m["f1_score"],
                    "roc_auc": m["roc_auc"],
                }
            )

        if 'AGGREGATED' in self.results:
            m = self.results['AGGREGATED']['metrics']
            records.append(
                {
                    "hospital": "AGGREGATED",
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1_score": m["f1_score"],
                    "roc_auc": m["roc_auc"],
                }
            )

        df_metrics = pd.DataFrame.from_records(records)
        df_long = df_metrics.melt(
            id_vars="hospital",
            var_name="metric",
            value_name="value",
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            data=df_long,
            x="hospital",
            y="value",
            hue="metric",
            ax=ax,
        )
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Metrics by Hospital")
        ax.legend(title="Metric")
        fig.tight_layout()

        out_path = self.output_dir / "figures" / "metrics_comparison.png"
        fig.savefig(out_path)
        plt.close(fig)

    def _plot_prediction_distribution(self):
        """Plot distribution of predicted probabilities for positives vs negatives (aggregated)."""
        if 'AGGREGATED' in self.results:
            y_true = self.results['AGGREGATED']['y_test']
            y_score = self.results['AGGREGATED']['y_pred_proba']
        else:
            y_true = np.concatenate([self.results[h]['y_test'] for h in HOSPITALS])
            y_score = np.concatenate([self.results[h]['y_pred_proba'] for h in HOSPITALS])

        df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
        df["label"] = df["y_true"].map({0: "Survived", 1: "Died"})

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=df,
            x="y_score",
            hue="label",
            bins=30,
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_xlabel("Predicted probability of mortality")
        ax.set_title("Prediction Probability Distribution (Aggregated)")
        fig.tight_layout()

        out_path = self.output_dir / "figures" / "prediction_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)

    def generate_report(self, metrics: List[Dict]):
        # unchanged body from your current file, saving JSON + MD
        # (no changes needed for MLflow here)
        # ...
        # (Use your full implementation from the current file)
        # (For brevity, not repeating; keep as-is)
        # -------------------------
        # BEGIN: your original implementation
        print("\n" + "="*60)
        print("Generating Evaluation Report")
        print("="*60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics as JSON
        json_path = self.output_dir / f'evaluation_metrics_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"JSON metrics saved to {json_path}")

        # Generate markdown report
        md_path = self.output_dir / f'evaluation_report_{timestamp}.md'
        with open(md_path, 'w') as f:
            f.write("# HIMAS Federated Model - Test Set Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model Path:** `{self.model_path}`\n\n")
            f.write("---\n\n")
            agg_metrics = metrics[-1]
            f.write("## Executive Summary\n\n")
            f.write(
                f"The federated learning model was evaluated on test data from three hospitals, "
                f"comprising a total of {agg_metrics['n_samples']} patient ICU stays. "
                f"The model achieved an overall accuracy of {agg_metrics['accuracy']:.2%} "
                f"with an ROC AUC of {agg_metrics['roc_auc']:.3f}, demonstrating strong discriminative "
                f"ability for ICU mortality prediction.\n\n"
            )
            f.write("## Performance by Hospital\n\n")
            for m in metrics[:-1]:
                f.write(f"### {m['hospital'].replace('_', ' ').title()}\n\n")
                f.write(f"- **Test Samples:** {m['n_samples']:,}\n")
                f.write(f"- **Mortality Prevalence:** {m['prevalence']:.2%}\n")
                f.write(f"- **Accuracy:** {m['accuracy']:.2%}\n")
                f.write(f"- **Precision:** {m['precision']:.2%}\n")
                f.write(f"- **Recall (Sensitivity):** {m['recall']:.2%}\n")
                f.write(f"- **Specificity:** {m['specificity']:.2%}\n")
                f.write(f"- **F1 Score:** {m['f1_score']:.3f}\n")
                f.write(f"- **ROC AUC:** {m['roc_auc']:.3f}\n")
                f.write(f"- **Average Precision:** {m['average_precision']:.3f}\n\n")
                cm = m['confusion_matrix']
                f.write("**Confusion Matrix:**\n")
                f.write(f"- True Negatives: {cm['true_negatives']:,}\n")
                f.write(f"- False Positives: {cm['false_positives']:,}\n")
                f.write(f"- False Negatives: {cm['false_negatives']:,}\n")
                f.write(f"- True Positives: {cm['true_positives']:,}\n\n")

            # Aggregated Results
            f.write("## Aggregated Performance Across All Hospitals\n\n")
            f.write(f"- **Total Test Samples:** {agg_metrics['n_samples']:,}\n")
            f.write(f"- **Overall Mortality Prevalence:** {agg_metrics['prevalence']:.2%}\n")
            f.write(f"- **Accuracy:** {agg_metrics['accuracy']:.2%}\n")
            f.write(f"- **Precision (PPV):** {agg_metrics['precision']:.2%}\n")
            f.write(f"- **Recall (Sensitivity):** {agg_metrics['recall']:.2%}\n")
            f.write(f"- **Specificity:** {agg_metrics['specificity']:.2%}\n")
            f.write(f"- **NPV:** {agg_metrics['npv']:.2%}\n")
            f.write(f"- **F1 Score:** {agg_metrics['f1_score']:.3f}\n")
            f.write(f"- **ROC AUC:** {agg_metrics['roc_auc']:.3f}\n")
            f.write(f"- **Average Precision:** {agg_metrics['average_precision']:.3f}\n\n")
            f.write("## Clinical Interpretation\n\n")
            f.write(
                f"The model demonstrates strong performance with a recall of {agg_metrics['recall']:.2%}, "
                f"indicating it successfully identifies {agg_metrics['recall']:.2%} of patients who will "
                f"experience ICU mortality. The precision of {agg_metrics['precision']:.2%} suggests that "
                f"when the model predicts mortality, it is correct {agg_metrics['precision']:.2%} of the time. "
                f"The high specificity of {agg_metrics['specificity']:.2%} indicates the model rarely "
                f"generates false alarms for patients who will survive.\n\n"
            )
            f.write("The ROC AUC of {:.3f} demonstrates excellent discriminative ability, ".format(
                agg_metrics['roc_auc']))
            f.write(
                "substantially exceeding the performance of random prediction (AUC = 0.5). "
                "This suggests the federated learning approach successfully learned meaningful patterns "
                "across the three hospitals without requiring centralized patient data.\n\n"
            )
            f.write("## Visualizations\n\n")
            f.write("The following visualizations are available in the `figures/` directory:\n\n")
            f.write("1. **ROC Curves**\n2. **Precision-Recall Curves**\n3. **Confusion Matrices**\n"
                    "4. **Metrics Comparison**\n5. **Prediction Distribution**\n\n")
            f.write("---\n\n*Report generated automatically by HIMAS Model Evaluation System*\n")
        print(f"Markdown report saved to {md_path}")

        # Log artifacts to MLflow
        mlflow.log_artifact(str(json_path), artifact_path="evaluation")
        mlflow.log_artifact(str(md_path), artifact_path="evaluation")
        mlflow.log_artifacts(str(self.output_dir / "figures"), artifact_path="evaluation/figures")

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Test Samples: {metrics[-1]['n_samples']:,}")
        print(f"Accuracy: {metrics[-1]['accuracy']:.2%}")
        print(f"ROC AUC: {metrics[-1]['roc_auc']:.3f}")
        print(f"Precision: {metrics[-1]['precision']:.2%}")
        print(f"Recall: {metrics[-1]['recall']:.2%}")
        print(f"F1 Score: {metrics[-1]['f1_score']:.3f}")
        print("="*60 + "\n")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("HIMAS FEDERATED MODEL EVALUATION")
    print("="*60)
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Model: {MODEL_PATH}")
    print("="*60 + "\n")

    # Wrap the whole evaluation in a single MLflow run
    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tags({"role": "evaluation"})

        evaluator = ModelEvaluator(MODEL_PATH, PROJECT_ID)
        evaluator.load_model()
        metrics = evaluator.evaluate_all_hospitals()
        evaluator.generate_visualizations()
        evaluator.generate_report(metrics)

        # Log model path param for traceability
        mlflow.log_param("evaluated_model_path", MODEL_PATH)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("Results saved to: evaluation_results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
