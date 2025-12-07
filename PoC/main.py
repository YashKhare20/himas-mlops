"""
Cloud Function to monitor deployed model performance.
"""

import json
import requests
from google.cloud import bigquery
from datetime import datetime
import functions_framework
import google.auth.transport.requests
from google.auth import compute_engine
import traceback

# Configuration
PROJECT_ID = "erudite-carving-472018-r5"
DATASET_ID = "federated"
SERVICE_URL = "https://himas-prediction-service-1089649594993.us-central1.run.app"
ACCURACY_THRESHOLD = 0.85
RECALL_THRESHOLD = 0.70
NUM_TEST_SAMPLES = 100

def store_monitoring_results(metrics):
    """Store monitoring results in BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.production_monitoring.model_performance"
    
    row = {
        "timestamp": metrics['timestamp'],
        "samples_tested": metrics['samples_tested'],
        "accuracy": metrics['accuracy'],
        "recall": metrics['recall'],
        "precision": metrics['precision'],
        "tp": metrics['tp'],
        "fp": metrics['fp'],
        "fn": metrics['fn'],
        "tn": metrics['tn']
    }
    
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        print(f"Error storing results: {errors}")
    else:
        print(f"✅ Results stored in BigQuery")

@functions_framework.http
def monitor_model(request):
    """Main entry point."""
    try:
        print("="*70)
        print("Model Performance Monitoring")
        print("="*70)
        
        # Get ID token
        auth_req = google.auth.transport.requests.Request()
        credentials = compute_engine.IDTokenCredentials(
            auth_req, SERVICE_URL, use_metadata_identity_endpoint=True
        )
        credentials.refresh(auth_req)
        token = credentials.token
        
        # Query BigQuery
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.hospital_a_data`
        WHERE data_split = 'test'
        ORDER BY RAND()
        LIMIT {NUM_TEST_SAMPLES}
        """
        
        results = client.query(query).result()
        
        # Test predictions
        correct = 0
        total = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        
        for row in results:
            sample = dict(row.items())
            ground_truth = sample.pop('hospital_expire_flag')
            
            for col in ['subject_id', 'hadm_id', 'data_split', 'hospital', 'stay_id', 
                       'icu_intime', 'icu_outtime', 'deathtime', 'death_date', 
                       'assigned_hospital', 'icu_mortality_label']:
                sample.pop(col, None)
            
            for key, value in list(sample.items()):
                if hasattr(value, 'isoformat'):
                    sample[key] = value.isoformat()
            
            try:
                response = requests.post(
                    f"{SERVICE_URL}/predict",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=sample,
                    timeout=10
                )
                
                if response.status_code == 200:
                    prediction = response.json().get('prediction', 0)
                    if prediction == ground_truth:
                        correct += 1
                    
                    if ground_truth == 1 and prediction == 1:
                        tp += 1
                    elif ground_truth == 0 and prediction == 1:
                        fp += 1
                    elif ground_truth == 1 and prediction == 0:
                        fn += 1
                    else:
                        tn += 1
                    
                    total += 1
            except:
                continue
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'samples_tested': total,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        
        print(f"Results: {total} samples, Acc={accuracy:.2%}, Recall={recall:.2%}")
        
        # Store in BigQuery
        store_monitoring_results(metrics)
        
        # Check if retraining needed
        needs_retraining = False
        reasons = []
        
        if accuracy < ACCURACY_THRESHOLD:
            needs_retraining = True
            reasons.append(f"Accuracy ({accuracy:.2%}) < {ACCURACY_THRESHOLD:.2%}")
        
        if recall < RECALL_THRESHOLD:
            needs_retraining = True
            reasons.append(f"Recall ({recall:.2%}) < {RECALL_THRESHOLD:.2%}")
        
        if needs_retraining:
            print(f"⚠️  ALERT: Retraining recommended - {', '.join(reasons)}")
            return json.dumps({
                'status': 'alert_retraining_needed',
                'reasons': reasons,
                'metrics': metrics
            })
        else:
            print("✅ Model performance acceptable")
            return json.dumps({'status': 'ok', 'metrics': metrics})
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        return json.dumps({'error': str(e)}), 500
