from google.cloud import bigquery
import json
from datetime import datetime

client = bigquery.Client(project="erudite-carving-472018-r5")

# Query real test data
query = """
SELECT *
FROM `erudite-carving-472018-r5.federated.hospital_a_data`
WHERE data_split = 'test'
LIMIT 5
"""

results = client.query(query).result()

print("Real test samples from BigQuery:\n")
for i, row in enumerate(results, 1):
    sample = {}
    
    # Convert row to dict, handling datetime
    for key, value in row.items():
        if isinstance(value, datetime):
            sample[key] = value.isoformat()
        else:
            sample[key] = value
    
    # Remove columns not used for prediction
    ground_truth = sample.pop('hospital_expire_flag', None)
    sample.pop('subject_id', None)
    sample.pop('hadm_id', None)
    sample.pop('data_split', None)
    sample.pop('hospital', None)
    
    print(f"Sample {i} (actual outcome: {ground_truth}):")
    print(json.dumps(sample))
    print()

