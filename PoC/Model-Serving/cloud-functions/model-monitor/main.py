"""
Cloud Function to monitor deployed model performance.
Automatically triggers retraining when performance degrades.
"""

import json
import requests
from google.cloud import bigquery
from google.cloud.devtools import cloudbuild_v1
from datetime import datetime
import functions_framework
import google.auth.transport.requests
from google.auth import compute_engine
import traceback

# Configuration
PROJECT_ID = "erudite-carving-472018-r5"
DATASET_ID = "federated"
SERVICE_URL = "https://himas-prediction-service-1089649594993.us-central1.run.app"
ACCURACY_THRESHOLD = 0.95
RECALL_THRESHOLD = 0.90
NUM_TEST_SAMPLES = 100
GITHUB_REPO = "manjushamg26/himas-mlops"
GITHUB_BRANCH = "main"

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

def trigger_cloud_build():
    """Automatically trigger Cloud Build to retrain model."""
    try:
        print("\n🚀 Triggering Cloud Build for automatic retraining...")
        
        client = cloudbuild_v1.CloudBuildClient()
        
        # Create build configuration
        build = cloudbuild_v1.Build()
        
        # Use GitHub source
        build.source = cloudbuild_v1.Source()
        build.source.repo_source = cloudbuild_v1.RepoSource()
        build.source.repo_source.project_id = PROJECT_ID
        build.source.repo_source.repo_name = f"github_{GITHUB_REPO.replace('/', '_')}"
        build.source.repo_source.branch_name = GITHUB_BRANCH
        
        # Trigger the build
        operation = client.create_build(
            project_id=PROJECT_ID,
            build=build
        )
        
        build_id = operation.metadata.build.id
        print(f"✅ Cloud Build triggered successfully!")
        print(f"   Build ID: {build_id}")
        print(f"   View: https://console.cloud.google.com/cloud-build/builds/{build_id}")
        
        return build_id
        
    except Exception as e:
        print(f"❌ Failed to trigger Cloud Build: {e}")
        print(traceback.format_exc())
        return None

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
        
        print(f"\nResults: {total} samples tested")
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
            print("\n" + "="*70)
            print("⚠️  MODEL DEGRADATION DETECTED - TRIGGERING RETRAINING")
            print("="*70)
            for reason in reasons:
                print(f"  • {reason}")
            
            # Automatically trigger Cloud Build
            build_id = trigger_cloud_build()
            
            if build_id:
                return json.dumps({
                    'status': 'retraining_triggered_automatically',
                    'reasons': reasons,
                    'metrics': metrics,
                    'build_id': build_id,
                    'build_url': f"https://console.cloud.google.com/cloud-build/builds/{build_id}"
                })
            else:
                return json.dumps({
                    'status': 'alert_retraining_failed_to_trigger',
                    'reasons': reasons,
                    'metrics': metrics
                }), 500
        else:
            print("\n✅ Model performance acceptable - no retraining needed")
            return json.dumps({
                'status': 'ok',
                'metrics': metrics
            })
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        return json.dumps({'error': str(e)}), 500
