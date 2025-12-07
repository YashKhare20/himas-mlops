"""
Cloud Function to monitor deployed model performance.
Tests model periodically and triggers retraining if performance degrades.
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
    try:
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
            print(f"❌ Error storing results: {errors}")
            return False
        else:
            print(f"✅ Results stored in BigQuery")
            return True
    except Exception as e:
        print(f"❌ Storage error: {e}")
        return False

@functions_framework.http
def monitor_model(request):
    """Main entry point - monitors model and triggers retraining if needed."""
    try:
        print("="*70)
        print("Model Performance Monitoring")
        print("="*70)
        
        # Get ID token for Cloud Run
        auth_req = google.auth.transport.requests.Request()
        credentials = compute_engine.IDTokenCredentials(
            auth_req, SERVICE_URL, use_metadata_identity_endpoint=True
        )
        credentials.refresh(auth_req)
        token = credentials.token
        
        # Query test data from BigQuery
        print(f"Querying {NUM_TEST_SAMPLES} test samples...")
        client = bigquery.Client(project=PROJECT_ID)
        
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.hospital_a_data`
        WHERE data_split = 'test'
        ORDER BY RAND()
        LIMIT {NUM_TEST_SAMPLES}
        """
        
        results = client.query(query).result()
        
        # Make predictions and track results
        correct = 0
        total = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        
        for row in results:
            sample = dict(row.items())
            ground_truth = sample.pop('hospital_expire_flag')
            
            # Remove non-feature columns
            for col in ['subject_id', 'hadm_id', 'data_split', 'hospital', 'stay_id', 
                       'icu_intime', 'icu_outtime', 'deathtime', 'death_date', 
                       'assigned_hospital', 'icu_mortality_label']:
                sample.pop(col, None)
            
            # Convert datetime to string
            for key, value in list(sample.items()):
                if hasattr(value, 'isoformat'):
                    sample[key] = value.isoformat()
            
            try:
                response = requests.post(
                    f"{SERVICE_URL}/predict",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json"
                    },
                    json=sample,
                    timeout=10
                )
                
                if response.status_code == 200:
                    prediction = response.json().get('prediction', 0)
                    
                    if prediction == ground_truth:
                        correct += 1
                    
                    # Confusion matrix
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
        
        # Calculate comprehensive metrics
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
        
        print(f"Results: {total} samples tested")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  Precision: {precision:.2%}")
        
        # Store results in BigQuery
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
            print("="*70)
            print("⚠️  ALERT: RETRAINING RECOMMENDED")
            print("="*70)
            for reason in reasons:
                print(f"  - {reason}")
            print("\nAction required: Push to GitHub to trigger retraining pipeline")
            print("="*70)
            
            return json.dumps({
                'status': 'alert_retraining_needed',
                'reasons': reasons,
                'metrics': metrics
            })
        else:
            print("✅ Model performance acceptable - no retraining needed")
            return json.dumps({
                'status': 'ok',
                'metrics': metrics
            })
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        return json.dumps({'error': str(e)}), 500
