import json
import sys
from pathlib import Path

def detect_bias(results_dir, acc_threshold=0.05, recall_threshold=0.07):
    """
    Check if any hospital has significantly different metrics from aggregate.
    Returns True if bias detected, False otherwise.
    """
    # Load evaluation results
    results_file = list(Path(results_dir).glob("*.json"))[0]
    with open(results_file) as f:
        results = json.load(f)
    
    # Get aggregated metrics
    agg = next(r for r in results if r["hospital"] == "AGGREGATED")
    agg_acc = agg["accuracy"]
    agg_rec = agg["recall"]
    
    # Check each hospital
    bias_detected = False
    for hospital in ["hospital_a", "hospital_b", "hospital_c"]:
        h_metrics = next(r for r in results if r["hospital"] == hospital)
        h_acc = h_metrics["accuracy"]
        h_rec = h_metrics["recall"]
        
        acc_diff = abs(agg_acc - h_acc)
        rec_diff = abs(agg_rec - h_rec)
        
        print(f"{hospital}: Acc diff={acc_diff:.4f}, Recall diff={rec_diff:.4f}")
        
        if acc_diff > acc_threshold or rec_diff > recall_threshold:
            print(f"❌ Bias detected in {hospital}!")
            bias_detected = True
    
    return bias_detected

if __name__ == "__main__":
    bias_found = detect_bias("evaluation_results")
    if bias_found:
        sys.exit(1)  # Fail the build
    else:
        print("✅ No bias detected")
        sys.exit(0)  # Success