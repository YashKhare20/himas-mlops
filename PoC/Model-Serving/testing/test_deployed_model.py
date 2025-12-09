"""
Test deployed Cloud Run model with real test data from BigQuery.

This script:
1. Loads real test samples from BigQuery (same data used in evaluation)
2. Sends them to the deployed Cloud Run endpoint
3. Compares predictions to ground truth
4. Reports accuracy metrics
"""

import json
import subprocess
from google.cloud import bigquery
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Configuration
PROJECT_ID = "erudite-carving-472018-r5"
DATASET_ID = "federated"
SERVICE_URL = "https://himas-prediction-service-1089649594993.us-central1.run.app"
NUM_SAMPLES = 50  # Number of test samples to try

def get_auth_token() -> str:
    """Get Google Cloud auth token for API calls."""
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()

def convert_value(value):
    """Convert BigQuery types to JSON-serializable types."""
    from datetime import datetime, date
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, bytes):
        return value.decode('utf-8')
    elif pd.isna(value):
        return None
    return value

def load_test_samples(n_samples: int) -> List[Tuple[Dict, int]]:
    """Load real test samples from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.hospital_a_data`
    WHERE data_split = 'test'
    LIMIT {n_samples}
    """
    
    results = client.query(query).result()
    
    samples = []
    for row in results:
        # Convert to dict
        sample = {key: convert_value(value) for key, value in row.items()}
        
        # Extract ground truth
        ground_truth = sample.pop('hospital_expire_flag')
        
        # Remove non-feature columns
        for col in ['subject_id', 'hadm_id', 'data_split', 'hospital']:
            sample.pop(col, None)
        
        samples.append((sample, ground_truth))
    
    print(f"✅ Loaded {len(samples)} test samples from BigQuery")
    return samples

def make_prediction(sample: Dict, token: str) -> Dict:
    """Send prediction request to Cloud Run service."""
    import requests
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{SERVICE_URL}/predict",
        headers=headers,
        json=sample,
        timeout=30
    )
    
    response.raise_for_status()
    return response.json()

def main():
    print("="*70)
    print("Testing Deployed HIMAS Model on Real Test Data")
    print("="*70)
    print(f"Service: {SERVICE_URL}")
    print(f"Dataset: {PROJECT_ID}.{DATASET_ID}")
    print(f"Samples: {NUM_SAMPLES}")
    print("="*70)
    print()
    
    # Load test samples
    print("Loading test samples from BigQuery...")
    samples = load_test_samples(NUM_SAMPLES)
    
    # Get auth token
    print("Getting authentication token...")
    token = get_auth_token()
    
    # Make predictions
    print(f"\nMaking predictions on {len(samples)} samples...")
    results = {
        'correct': 0,
        'incorrect': 0,
        'predictions': [],
        'by_outcome': {0: {'correct': 0, 'total': 0}, 1: {'correct': 0, 'total': 0}}
    }
    
    for i, (sample, ground_truth) in enumerate(samples, 1):
        try:
            pred_response = make_prediction(sample, token)
            prediction = pred_response.get('prediction', 0)
            probability = pred_response.get('probability', 0.0)
            
            is_correct = (prediction == ground_truth)
            
            if is_correct:
                results['correct'] += 1
            else:
                results['incorrect'] += 1
            
            # Track by outcome
            results['by_outcome'][ground_truth]['total'] += 1
            if is_correct:
                results['by_outcome'][ground_truth]['correct'] += 1
            
            results['predictions'].append({
                'sample': i,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'probability': probability,
                'correct': is_correct
            })
            
            # Print progress
            if i % 10 == 0:
                acc = results['correct'] / i
                print(f"  [{i}/{len(samples)}] Accuracy so far: {acc:.2%}")
                
        except Exception as e:
            print(f"  ❌ Error on sample {i}: {e}")
            continue
    
    # Calculate final metrics
    total = results['correct'] + results['incorrect']
    accuracy = results['correct'] / total if total > 0 else 0
    
    # Calculate recall and precision
    survived_correct = results['by_outcome'][0]['correct']
    survived_total = results['by_outcome'][0]['total']
    deceased_correct = results['by_outcome'][1]['correct']
    deceased_total = results['by_outcome'][1]['total']
    
    recall = deceased_correct / deceased_total if deceased_total > 0 else 0
    
    # Count true positives, false positives
    tp = deceased_correct
    fp = survived_total - survived_correct
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total Samples: {total}")
    print(f"Correct: {results['correct']} ({accuracy:.2%})")
    print(f"Incorrect: {results['incorrect']} ({1-accuracy:.2%})")
    print()
    print(f"Survived (0): {survived_correct}/{survived_total} correct")
    print(f"Deceased (1): {deceased_correct}/{deceased_total} correct")
    print()
    print(f"Recall (Sensitivity): {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print("="*70)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"deployed_model_test_{timestamp}.json"
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'service_url': SERVICE_URL,
        'num_samples': total,
        'metrics': {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'correct': results['correct'],
            'incorrect': results['incorrect']
        },
        'by_outcome': results['by_outcome'],
        'predictions': results['predictions']
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return accuracy

if __name__ == "__main__":
    try:
        import pandas as pd
        import requests
        accuracy = main()
        sys.exit(0 if accuracy > 0.7 else 1)  # Exit with error if accuracy < 70%
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
