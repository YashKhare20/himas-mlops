"""
Bias Checker - CI/CD Gate
=========================

Checks if model passes bias thresholds. Used as a gate in CI/CD pipeline.
Returns exit code 0 if pass, 1 if fail (blocks deployment).

Usage:
    python bias_detection/bias_checker.py --bias-summary bias_detection_results/reports/bias_summary.json
"""

import argparse
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def check_bias_thresholds(
    bias_summary_path: Path,
    dp_threshold: float = 0.1,
    eo_threshold: float = 0.1
) -> int:
    """
    Check if model passes bias thresholds.
    
    Args:
        bias_summary_path: Path to bias_summary.json
        dp_threshold: Maximum acceptable demographic parity difference
        eo_threshold: Maximum acceptable equalized odds difference
        
    Returns:
        Exit code: 0 if pass, 1 if fail
    """
    if not bias_summary_path.exists():
        logger.error(f"Bias summary not found: {bias_summary_path}")
        return 1
    
    with open(bias_summary_path) as f:
        bias_data = json.load(f)
    
    logger.info("="*70)
    logger.info("BIAS CHECK - CI/CD GATE")
    logger.info("="*70)
    
    # Check if already marked as passed/failed
    if 'bias_check_passed' in bias_data:
        passed = bias_data['bias_check_passed']
        max_dp = bias_data.get('max_demographic_parity', 0)
        max_eo = bias_data.get('max_equalized_odds', 0)
        
        if passed:
            logger.info("✅ BIAS CHECK PASSED")
            logger.info(f"  Max Demographic Parity: {max_dp:.4f} (threshold: {dp_threshold})")
            logger.info(f"  Max Equalized Odds: {max_eo:.4f} (threshold: {eo_threshold})")
            logger.info("  Model deployment approved.")
            logger.info("="*70)
            return 0
        else:
            logger.error("❌ BIAS CHECK FAILED")
            logger.error(f"  Max Demographic Parity: {max_dp:.4f} (threshold: {dp_threshold})")
            logger.error(f"  Max Equalized Odds: {max_eo:.4f} (threshold: {eo_threshold})")
            
            if 'violations' in bias_data:
                logger.error("\nBias Violations:")
                for violation in bias_data['violations']:
                    logger.error(
                        f"  {violation['feature']} - {violation['metric']}: "
                        f"{violation['value']:.4f} > {violation['threshold']:.4f} "
                        f"({violation['severity']} severity)"
                    )
            
            logger.error("\n🚫 Model deployment BLOCKED due to bias violations.")
            logger.error("="*70)
            return 1
    
    # Fallback: Calculate from fairness metrics
    fairness_metrics = bias_data.get('fairness_metrics', {})
    if not fairness_metrics:
        logger.error("No fairness metrics found in bias summary")
        return 1
    
    # Get max values across all features
    dp_values = []
    eo_values = []
    
    for feature, metrics in fairness_metrics.items():
        if 'demographic_parity_difference' in metrics:
            dp_values.append(abs(metrics['demographic_parity_difference']))
        if 'equalized_odds_difference' in metrics:
            eo_values.append(abs(metrics['equalized_odds_difference']))
    
    max_dp = max(dp_values) if dp_values else 0
    max_eo = max(eo_values) if eo_values else 0
    
    # Check thresholds
    passed = (max_dp <= dp_threshold) and (max_eo <= eo_threshold)
    
    if passed:
        logger.info("✅ BIAS CHECK PASSED")
        logger.info(f"  Max Demographic Parity: {max_dp:.4f} (threshold: {dp_threshold})")
        logger.info(f"  Max Equalized Odds: {max_eo:.4f} (threshold: {eo_threshold})")
        logger.info("  Model deployment approved.")
        logger.info("="*70)
        return 0
    else:
        logger.error("❌ BIAS CHECK FAILED")
        logger.error(f"  Max Demographic Parity: {max_dp:.4f} (threshold: {dp_threshold})")
        logger.error(f"  Max Equalized Odds: {max_eo:.4f} (threshold: {eo_threshold})")
        logger.error("\n🚫 Model deployment BLOCKED due to bias violations.")
        logger.error("="*70)
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check if model passes bias thresholds (CI/CD gate)'
    )
    parser.add_argument(
        '--bias-summary',
        type=str,
        required=True,
        help='Path to bias_summary.json'
    )
    parser.add_argument(
        '--threshold-demographic-parity',
        type=float,
        default=0.1,
        help='Demographic parity threshold (default: 0.1)'
    )
    parser.add_argument(
        '--threshold-equalized-odds',
        type=float,
        default=0.1,
        help='Equalized odds threshold (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    exit_code = check_bias_thresholds(
        Path(args.bias_summary),
        args.threshold_demographic_parity,
        args.threshold_equalized_odds
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

