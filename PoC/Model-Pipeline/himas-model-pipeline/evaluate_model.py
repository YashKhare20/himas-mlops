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

# Configuration
PROJECT_ID = "erudite-carving-472018-r5"
DATASET_ID = "federated"
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
            metrics = self.evaluate_hospital(hospital)
            hospital_metrics.append(metrics)

        # Compute aggregated metrics across all hospitals
        print("\n" + "="*60)
        print("Computing Aggregated Metrics Across All Hospitals")
        print("="*60)

        all_y_test = np.concatenate(
            [self.results[h]['y_test'] for h in HOSPITALS])
        all_y_pred = np.concatenate(
            [self.results[h]['y_pred'] for h in HOSPITALS])
        all_y_pred_proba = np.concatenate(
            [self.results[h]['y_pred_proba'] for h in HOSPITALS])

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

        aggregated_metrics['specificity'] = float(
            tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        aggregated_metrics['npv'] = float(
            tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        aggregated_metrics['ppv'] = aggregated_metrics['precision']

        hospital_metrics.append(aggregated_metrics)

        return hospital_metrics

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

    def _plot_roc_curves(self):
        """Plot ROC curves for all hospitals."""
        plt.figure(figsize=(10, 8))

        for hospital in HOSPITALS:
            data = self.results[hospital]
            fpr, tpr, _ = roc_curve(data['y_test'], data['y_pred_proba'])
            auc = roc_auc_score(data['y_test'], data['y_pred_proba'])
            plt.plot(
                fpr, tpr, label=f'{hospital.replace("_", " ").title()} (AUC = {auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - ICU Mortality Prediction by Hospital',
                  fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' /
                    'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("ROC curves saved")

    def _plot_precision_recall_curves(self):
        """Plot Precision-Recall curves for all hospitals."""
        plt.figure(figsize=(10, 8))

        for hospital in HOSPITALS:
            data = self.results[hospital]
            precision, recall, _ = precision_recall_curve(
                data['y_test'], data['y_pred_proba'])
            ap = average_precision_score(data['y_test'], data['y_pred_proba'])
            plt.plot(recall, precision,
                     label=f'{hospital.replace("_", " ").title()} (AP = {ap:.3f})', linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - ICU Mortality Prediction by Hospital',
                  fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' /
                    'precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Precision-Recall curves saved")

    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all hospitals."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, hospital in enumerate(HOSPITALS):
            data = self.results[hospital]
            cm = confusion_matrix(data['y_test'], data['y_pred'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Survived', 'Deceased'],
                        yticklabels=['Survived', 'Deceased'])
            axes[idx].set_title(
                f'{hospital.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' /
                    'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Confusion matrices saved")

    def _plot_metrics_comparison(self):
        """Plot comparison of key metrics across hospitals."""
        metrics_data = []
        for hospital in HOSPITALS:
            m = self.results[hospital]['metrics']
            metrics_data.append({
                'Hospital': hospital.replace('_', ' ').title(),
                'Accuracy': m['accuracy'],
                'Precision': m['precision'],
                'Recall': m['recall'],
                'F1 Score': m['f1_score'],
                'ROC AUC': m['roc_auc']
            })

        df_metrics = pd.DataFrame(metrics_data)

        # Melt for grouped bar plot
        df_melted = df_metrics.melt(
            id_vars='Hospital', var_name='Metric', value_name='Score')

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melted, x='Metric', y='Score',
                    hue='Hospital', palette='Set2')
        plt.title('Performance Metrics Comparison Across Hospitals',
                  fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(title='Hospital', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' /
                    'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Metrics comparison saved")

    def _plot_prediction_distribution(self):
        """Plot distribution of prediction probabilities."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, hospital in enumerate(HOSPITALS):
            data = self.results[hospital]

            # Separate probabilities by actual outcome
            proba_survived = data['y_pred_proba'][data['y_test'] == 0]
            proba_deceased = data['y_pred_proba'][data['y_test'] == 1]

            axes[idx].hist(proba_survived, bins=30, alpha=0.6,
                           label='Survived (True)', color='green', density=True)
            axes[idx].hist(proba_deceased, bins=30, alpha=0.6,
                           label='Deceased (True)', color='red', density=True)
            axes[idx].axvline(x=0.5, color='black',
                              linestyle='--', linewidth=1, label='Threshold')
            axes[idx].set_title(
                f'{hospital.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(
                'Predicted Mortality Probability', fontsize=10)
            axes[idx].set_ylabel('Density', fontsize=10)
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' /
                    'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Prediction distribution saved")

    def generate_report(self, metrics: List[Dict]):
        """
        Generate comprehensive evaluation report.

        Args:
            metrics: List of metric dictionaries for each hospital
        """
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
            f.write(
                f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Model Path:** `{self.model_path}`\n\n")
            f.write("---\n\n")

            # Executive Summary
            agg_metrics = metrics[-1]  # Last entry is aggregated
            f.write("## Executive Summary\n\n")
            f.write(
                f"The federated learning model was evaluated on test data from three hospitals, ")
            f.write(
                f"comprising a total of {agg_metrics['n_samples']} patient ICU stays. ")
            f.write(
                f"The model achieved an overall accuracy of {agg_metrics['accuracy']:.2%} ")
            f.write(
                f"with an ROC AUC of {agg_metrics['roc_auc']:.3f}, demonstrating strong discriminative ")
            f.write(f"ability for ICU mortality prediction.\n\n")

            # Hospital-Specific Results
            f.write("## Performance by Hospital\n\n")
            for m in metrics[:-1]:  # Exclude aggregated
                f.write(f"### {m['hospital'].replace('_', ' ').title()}\n\n")
                f.write(f"- **Test Samples:** {m['n_samples']:,}\n")
                f.write(f"- **Mortality Prevalence:** {m['prevalence']:.2%}\n")
                f.write(f"- **Accuracy:** {m['accuracy']:.2%}\n")
                f.write(f"- **Precision:** {m['precision']:.2%}\n")
                f.write(f"- **Recall (Sensitivity):** {m['recall']:.2%}\n")
                f.write(f"- **Specificity:** {m['specificity']:.2%}\n")
                f.write(f"- **F1 Score:** {m['f1_score']:.3f}\n")
                f.write(f"- **ROC AUC:** {m['roc_auc']:.3f}\n")
                f.write(
                    f"- **Average Precision:** {m['average_precision']:.3f}\n\n")

                cm = m['confusion_matrix']
                f.write(f"**Confusion Matrix:**\n")
                f.write(f"- True Negatives: {cm['true_negatives']:,}\n")
                f.write(f"- False Positives: {cm['false_positives']:,}\n")
                f.write(f"- False Negatives: {cm['false_negatives']:,}\n")
                f.write(f"- True Positives: {cm['true_positives']:,}\n\n")

            # Aggregated Results
            f.write("## Aggregated Performance Across All Hospitals\n\n")
            f.write(
                f"- **Total Test Samples:** {agg_metrics['n_samples']:,}\n")
            f.write(
                f"- **Overall Mortality Prevalence:** {agg_metrics['prevalence']:.2%}\n")
            f.write(f"- **Accuracy:** {agg_metrics['accuracy']:.2%}\n")
            f.write(f"- **Precision (PPV):** {agg_metrics['precision']:.2%}\n")
            f.write(
                f"- **Recall (Sensitivity):** {agg_metrics['recall']:.2%}\n")
            f.write(f"- **Specificity:** {agg_metrics['specificity']:.2%}\n")
            f.write(f"- **NPV:** {agg_metrics['npv']:.2%}\n")
            f.write(f"- **F1 Score:** {agg_metrics['f1_score']:.3f}\n")
            f.write(f"- **ROC AUC:** {agg_metrics['roc_auc']:.3f}\n")
            f.write(
                f"- **Average Precision:** {agg_metrics['average_precision']:.3f}\n\n")

            # Clinical Interpretation
            f.write("## Clinical Interpretation\n\n")
            f.write(
                f"The model demonstrates strong performance with a recall of {agg_metrics['recall']:.2%}, ")
            f.write(
                f"indicating it successfully identifies {agg_metrics['recall']:.2%} of patients who will ")
            f.write(
                f"experience ICU mortality. The precision of {agg_metrics['precision']:.2%} suggests that ")
            f.write(
                f"when the model predicts mortality, it is correct {agg_metrics['precision']:.2%} of the time. ")
            f.write(
                f"The high specificity of {agg_metrics['specificity']:.2%} indicates the model rarely ")
            f.write(f"generates false alarms for patients who will survive.\n\n")

            f.write("The ROC AUC of {:.3f} demonstrates excellent discriminative ability, ".format(
                agg_metrics['roc_auc']))
            f.write(
                "substantially exceeding the performance of random prediction (AUC = 0.5). ")
            f.write(
                "This suggests the federated learning approach successfully learned meaningful patterns ")
            f.write(
                "across the three hospitals without requiring centralized patient data.\n\n")

            # Visualizations
            f.write("## Visualizations\n\n")
            f.write(
                "The following visualizations are available in the `figures/` directory:\n\n")
            f.write(
                "1. **ROC Curves** - Receiver Operating Characteristic curves for each hospital\n")
            f.write(
                "2. **Precision-Recall Curves** - Performance across different decision thresholds\n")
            f.write(
                "3. **Confusion Matrices** - Classification outcomes for each hospital\n")
            f.write(
                "4. **Metrics Comparison** - Side-by-side comparison of key performance indicators\n")
            f.write(
                "5. **Prediction Distribution** - Distribution of predicted probabilities by outcome\n\n")

            f.write("---\n\n")
            f.write(
                "*Report generated automatically by HIMAS Model Evaluation System*\n")

        print(f"Markdown report saved to {md_path}")

        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Test Samples: {agg_metrics['n_samples']:,}")
        print(f"Accuracy: {agg_metrics['accuracy']:.2%}")
        print(f"ROC AUC: {agg_metrics['roc_auc']:.3f}")
        print(f"Precision: {agg_metrics['precision']:.2%}")
        print(f"Recall: {agg_metrics['recall']:.2%}")
        print(f"F1 Score: {agg_metrics['f1_score']:.3f}")
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

    # Initialize evaluator
    evaluator = ModelEvaluator(MODEL_PATH, PROJECT_ID)

    # Load model
    evaluator.load_model()

    # Evaluate on all hospitals
    metrics = evaluator.evaluate_all_hospitals()

    # Generate visualizations
    evaluator.generate_visualizations()

    # Generate comprehensive report
    evaluator.generate_report(metrics)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {evaluator.output_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
