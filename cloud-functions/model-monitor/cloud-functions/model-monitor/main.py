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
NUM_TEST_SAMPLES = 10

@functions_framework.http
def monitor_model(request):
    """Main entry point."""
    try:
        print("Starting Model Performance Monitoring")
        
        # Get ID token for Cloud Run (not access token!)
        print("Getting ID token for Cloud Run...")
        auth_req = google.auth.transport.requests.Request()
        credentials = compute_engine.IDTokenCredentials(
            auth_req,
            SERVICE_URL,
            use_metadata_identity_endpoint=True
        )
        credentials.refresh(auth_req)
        token = credentials.token
        print("✅ Got ID token")
        
        # Query BigQuery
        print(f"Querying BigQuery for {NUM_TEST_SAMPLES} test samples...")
        client = bigquery.Client(project=PROJECT_ID)
        
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.hospital_a_data`
        WHERE data_split = 'test'
        ORDER BY RAND()
        LIMIT {NUM_TEST_SAMPLES}
        """
        
        results = client.query(query).result()
        print("✅ Query executed")
        
        # Test predictions
        correct = 0
        total = 0
        
        for row in results:
            sample = dict(row.items())
            ground_truth = sample.pop('hospital_expire_flag')
            
            # Remove non-feature columns
            for col in ['subject_id', 'hadm_id', 'data_split', 'hospital', 'stay_id', 
                       'icu_intime', 'icu_outtime', 'deathtime', 'death_date', 
                       'assigned_hospital', 'icu_mortality_label']:
                sample.pop(col, None)
            
            # Convert datetime values
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
                
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    prediction = response.json().get('prediction', 0)
                    if prediction == ground_truth:
                        correct += 1
                    total += 1
                else:
                    print(f"Prediction failed: {response.status_code} - {response.text}")
                    
            except Exception as pred_error:
                print(f"Prediction error: {pred_error}")
                continue
        
        print(f"✅ Tested {total} samples, {correct} correct")
        
        accuracy = correct / total if total > 0 else 0
        
        return json.dumps({
            'status': 'ok',
            'samples_tested': total,
            'correct': correct,
            'accuracy': accuracy
        })
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(traceback.format_exc())
        return json.dumps({'error': str(e)}), 500
