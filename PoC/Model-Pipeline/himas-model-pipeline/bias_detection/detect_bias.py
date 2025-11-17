"""
Bias Detection Script
====================

Detects bias in trained model using Fairlearn for demographic slicing.
Evaluates model fairness across gender, age, insurance, and race groups.

Usage:
    python bias_detection/detect_bias.py --model-path models/.../model.keras
"""

import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import keras
from google.cloud import bigquery
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score
)
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)

# Fix imports - add parent directory to path
import sys
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Now import modules
from bias_detection.utils import (
    load_test_data_with_demographics,
    get_config_value
)
from scripts.evaluate_model import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_model_and_preprocessor(model_path: Path, project_id: str) -> tuple:
    """
    Load trained model and fit preprocessor on training data.
    
    Args:
        model_path: Path to trained Keras model
        project_id: GCP project ID
        
    Returns:
        Tuple of (model, preprocessor)
    """
    logger.info("Loading model...")
    model = keras.models.load_model(str(model_path))
    logger.info(f"Model loaded: {model.count_params():,} parameters")
    
    logger.info("Fitting preprocessor on training data...")
    preprocessor = DataPreprocessor()
    
    # Load training data from all hospitals
    dataset_id = get_config_value('tool.himas.data.dataset-id', 'federated_demo')
    hospitals = get_config_value(
        'tool.himas.data.hospital-names',
        ['hospital_a', 'hospital_b', 'hospital_c']
    )
    train_split = get_config_value('tool.himas.data.train-split', 'train')
    client = bigquery.Client(project=project_id)
    
    all_train_data = []
    for hospital in hospitals:
        query = f"SELECT * FROM `{project_id}.{dataset_id}.{hospital}_data` WHERE data_split = '{train_split}'"
        df_train = client.query(query).to_dataframe()
        all_train_data.append(df_train)
    
    df_train_combined = pd.concat(all_train_data, ignore_index=True)
    preprocessor.fit(df_train_combined)
    logger.info("Preprocessor fitted on training data")
    
    return model, preprocessor


def generate_predictions(model, preprocessor, test_data: pd.DataFrame) -> np.ndarray:
    """
    Generate model predictions on test data.
    
    Args:
        model: Trained Keras model
        preprocessor: Fitted DataPreprocessor
        test_data: Test dataframe
        
    Returns:
        Array of predicted probabilities
    """
    logger.info("Generating predictions...")
    X_test, y_test = preprocessor.transform(test_data)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    logger.info(f"Predictions generated for {len(y_pred_proba):,} samples")
    return y_pred_proba


def calculate_bias_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    demographics: pd.DataFrame
) -> Dict:
    """
    Calculate bias metrics using Fairlearn.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        demographics: DataFrame with demographic features
        
    Returns:
        Dictionary with bias metrics and per-slice metrics
    """
    logger.info("Calculating bias metrics...")
    
    # Define metrics to calculate per slice
    metrics = {
        'accuracy': accuracy_score,
        'recall': recall_score,
        'precision': precision_score,
        'f1': f1_score
    }
    
    results = {
        'fairness_metrics': {},
        'metrics_by_slice': {},
        'performance_gaps': {}
    }
    
    # Calculate fairness metrics for each demographic feature
    for feature in ['gender', 'age_group', 'insurance', 'race']:
        if feature not in demographics.columns:
            continue
            
        logger.info(f"Analyzing bias by {feature}...")
        
        # Create MetricFrame for this feature
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=demographics[feature]
        )
        
        # Store per-slice metrics
        results['metrics_by_slice'][feature] = mf.by_group.to_dict('index')
        
        # Calculate fairness metrics
        dp_diff = demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=demographics[feature]
        )
        
        eo_diff = equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=demographics[feature]
        )
        
        results['fairness_metrics'][feature] = {
            'demographic_parity_difference': float(dp_diff),
            'equalized_odds_difference': float(eo_diff)
        }
        
        # Calculate performance gaps (max - min)
        slice_metrics = mf.by_group
        for metric_name in metrics.keys():
            gap_key = f"{feature}_{metric_name}_gap"
            if metric_name in slice_metrics.columns:
                max_val = slice_metrics[metric_name].max()
                min_val = slice_metrics[metric_name].min()
                results['performance_gaps'][gap_key] = float(max_val - min_val)
        
        logger.info(f"  Demographic Parity: {dp_diff:.4f}")
        logger.info(f"  Equalized Odds: {eo_diff:.4f}")
    
    return results


def check_bias_thresholds(
    fairness_metrics: Dict,
    dp_threshold: float = 0.1,
    eo_threshold: float = 0.1
) -> Dict:
    """
    Check if bias metrics exceed thresholds.
    
    Args:
        fairness_metrics: Dictionary of fairness metrics
        dp_threshold: Maximum acceptable demographic parity difference
        eo_threshold: Maximum acceptable equalized odds difference
        
    Returns:
        Dictionary with bias check results
    """
    violations = []
    max_dp = 0.0
    max_eo = 0.0
    
    for feature, metrics in fairness_metrics.items():
        dp = abs(metrics['demographic_parity_difference'])
        eo = abs(metrics['equalized_odds_difference'])
        
        max_dp = max(max_dp, dp)
        max_eo = max(max_eo, eo)
        
        if dp > dp_threshold:
            violations.append({
                'feature': feature,
                'metric': 'demographic_parity_difference',
                'value': dp,
                'threshold': dp_threshold,
                'severity': 'high' if dp > 2 * dp_threshold else 'medium'
            })
        
        if eo > eo_threshold:
            violations.append({
                'feature': feature,
                'metric': 'equalized_odds_difference',
                'value': eo,
                'threshold': eo_threshold,
                'severity': 'high' if eo > 2 * eo_threshold else 'medium'
            })
    
    passed = len(violations) == 0
    
    return {
        'bias_check_passed': passed,
        'max_demographic_parity': max_dp,
        'max_equalized_odds': max_eo,
        'thresholds': {
            'demographic_parity': dp_threshold,
            'equalized_odds': eo_threshold
        },
        'violations': violations
    }


def main():
    """Main entry point for bias detection."""
    parser = argparse.ArgumentParser(description='Detect bias in trained model')
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model (.keras file)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Prediction threshold for binary classification'
    )
    parser.add_argument(
        '--dp-threshold',
        type=float,
        default=0.1,
        help='Demographic parity threshold (default: 0.1)'
    )
    parser.add_argument(
        '--eo-threshold',
        type=float,
        default=0.1,
        help='Equalized odds threshold (default: 0.1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='bias_detection_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Get model path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Use latest model from config
        try:
            from scripts.evaluate_model import get_latest_model_path
            model_path = get_latest_model_path()
        except:
            # Fallback: find latest model in models directory
            models_dir = Path('models')
            model_files = list(models_dir.glob('**/*.keras'))
            if not model_files:
                raise FileNotFoundError("No model files found. Please specify --model-path")
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using latest model: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info("="*70)
    logger.info("BIAS DETECTION")
    logger.info("="*70)
    logger.info(f"Model: {model_path}")
    logger.info(f"Threshold: {args.threshold}")
    
    # Load configuration
    project_id = get_config_value('tool.himas.data.project-id')
    dataset_id = get_config_value('tool.himas.data.dataset-id', 'federated_demo')
    target_col = get_config_value('tool.himas.data.target-column', 'icu_mortality_label')
    
    # Load model and preprocessor
    model, preprocessor = load_model_and_preprocessor(model_path, project_id)
    
    # Load test data with demographics
    client = bigquery.Client(project=project_id)
    test_data, demographics = load_test_data_with_demographics(
        project_id, dataset_id, client
    )
    
    # Generate predictions
    y_pred_proba = generate_predictions(model, preprocessor, test_data)
    y_true = test_data[target_col].values
    y_pred = (y_pred_proba >= args.threshold).astype(int)
    
    # Calculate bias metrics
    bias_results = calculate_bias_metrics(y_true, y_pred, demographics)
    
    # Check thresholds
    bias_check = check_bias_thresholds(
        bias_results['fairness_metrics'],
        args.dp_threshold,
        args.eo_threshold
    )
    
    # Prepare results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'prediction_threshold': args.threshold,
        'total_samples': len(y_true),
        'positive_samples': int(y_true.sum()),
        'mortality_rate': float(y_true.mean()),
        **bias_results,
        'bias_check': bias_check
    }
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'reports').mkdir(exist_ok=True)
    
    # Save detailed report
    report_path = output_dir / 'reports' / f'bias_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed report saved: {report_path}")
    
    # Save summary for CI/CD
    summary = {
        'timestamp': results['timestamp'],
        'model_path': results['model_path'],
        'bias_check_passed': bias_check['bias_check_passed'],
        'fairness_metrics': {
            'demographic_parity_difference': {
                feature: metrics['demographic_parity_difference']
                for feature, metrics in bias_results['fairness_metrics'].items()
            },
            'equalized_odds_difference': {
                feature: metrics['equalized_odds_difference']
                for feature, metrics in bias_results['fairness_metrics'].items()
            }
        },
        'thresholds': bias_check['thresholds'],
        'violations': bias_check['violations'],
        'max_demographic_parity': bias_check['max_demographic_parity'],
        'max_equalized_odds': bias_check['max_equalized_odds']
    }
    
    summary_path = output_dir / 'reports' / 'bias_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")
    
    # Log results
    logger.info("="*70)
    logger.info("BIAS DETECTION RESULTS")
    logger.info("="*70)
    logger.info(f"Bias Check: {'PASSED' if bias_check['bias_check_passed'] else 'FAILED'}")
    logger.info(f"Max Demographic Parity: {bias_check['max_demographic_parity']:.4f}")
    logger.info(f"Max Equalized Odds: {bias_check['max_equalized_odds']:.4f}")
    
    if bias_check['violations']:
        logger.warning("Bias Violations Detected:")
        for violation in bias_check['violations']:
            logger.warning(
                f"  {violation['feature']} - {violation['metric']}: "
                f"{violation['value']:.4f} > {violation['threshold']:.4f} "
                f"({violation['severity']} severity)"
            )
    
    logger.info("="*70)
    
    return results


if __name__ == '__main__':
    main()

