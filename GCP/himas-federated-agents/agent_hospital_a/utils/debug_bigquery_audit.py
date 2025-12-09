#!/usr/bin/env python3
"""
BigQuery Audit Logging Diagnostic

This script tests each step of the BigQuery audit logging process
to identify exactly where the failure occurs.

Run from your agent directory:
    python debug_bigquery_audit.py
"""

import os
import sys
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'erudite-carving-472018-r5')
HOSPITAL_ID = os.getenv('HOSPITAL_ID', 'hospital_a')
AUDIT_DATASET = os.getenv('AUDIT_LOG_DATASET', 'audit_logs')
AUDIT_TABLE = f'{HOSPITAL_ID}_audit_log'

FULL_TABLE_ID = f"{PROJECT_ID}.{AUDIT_DATASET}.{AUDIT_TABLE}"


def print_step(num: int, title: str):
    print(f"\n{'='*60}")
    print(f"  STEP {num}: {title}")
    print('='*60)


def print_result(success: bool, message: str, details: str = None):
    icon = "✓" if success else "✗"
    status = "PASS" if success else "FAIL"
    print(f"\n  {icon} {status}: {message}")
    if details:
        for line in details.split('\n'):
            print(f"      {line}")


# ============================================================================
# STEP 1: Check BigQuery Client
# ============================================================================

def check_bigquery_client():
    print_step(1, "BigQuery Client Creation")
    
    try:
        from google.cloud import bigquery
        print_result(True, "google-cloud-bigquery is installed")
    except ImportError as e:
        print_result(False, "google-cloud-bigquery NOT installed", str(e))
        print("\n  FIX: pip install google-cloud-bigquery")
        return None
    
    try:
        client = bigquery.Client(project=PROJECT_ID)
        print_result(True, f"BigQuery client created for project: {PROJECT_ID}")
        return client
    except Exception as e:
        print_result(False, "Failed to create BigQuery client", str(e))
        print("\n  FIX: Check your GCP credentials")
        print("       Run: gcloud auth application-default login")
        return None


# ============================================================================
# STEP 2: Check Dataset Exists
# ============================================================================

def check_dataset(client):
    print_step(2, f"Check Dataset: {AUDIT_DATASET}")
    
    dataset_ref = f"{PROJECT_ID}.{AUDIT_DATASET}"
    
    try:
        dataset = client.get_dataset(dataset_ref)
        print_result(True, f"Dataset exists: {dataset_ref}")
        print(f"      Location: {dataset.location}")
        print(f"      Created: {dataset.created}")
        return True
    except Exception as e:
        print_result(False, f"Dataset NOT found: {dataset_ref}", str(e))
        
        print("\n  Creating dataset...")
        try:
            from google.cloud import bigquery
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"
            client.create_dataset(dataset)
            print_result(True, f"Dataset created: {dataset_ref}")
            return True
        except Exception as e2:
            print_result(False, "Failed to create dataset", str(e2))
            print("\n  FIX: Check IAM permissions - need roles/bigquery.dataEditor")
            return False


# ============================================================================
# STEP 3: Check Table Exists and Schema
# ============================================================================

def check_table(client):
    print_step(3, f"Check Table: {AUDIT_TABLE}")
    
    try:
        table = client.get_table(FULL_TABLE_ID)
        print_result(True, f"Table exists: {FULL_TABLE_ID}")
        print(f"      Rows: {table.num_rows}")
        print(f"      Size: {table.num_bytes} bytes")
        print(f"      Created: {table.created}")
        
        # Print schema
        print("\n  Schema:")
        for field in table.schema:
            print(f"      - {field.name}: {field.field_type} ({field.mode})")
        
        return table.schema
        
    except Exception as e:
        print_result(False, f"Table NOT found: {FULL_TABLE_ID}", str(e))
        return None


# ============================================================================
# STEP 4: Create Table if Missing
# ============================================================================

def create_table(client):
    print_step(4, "Create Audit Table")
    
    from google.cloud import bigquery
    
    schema = [
        bigquery.SchemaField("log_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("hospital_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("action", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("action_category", "STRING"),
        bigquery.SchemaField("event_timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("logged_at", "TIMESTAMP"),
        bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_role", "STRING"),
        bigquery.SchemaField("patient_age_bucket", "STRING"),
        bigquery.SchemaField("risk_score", "FLOAT64"),
        bigquery.SchemaField("risk_level", "STRING"),
        bigquery.SchemaField("target_hospital", "STRING"),
        bigquery.SchemaField("query_type", "STRING"),
        bigquery.SchemaField("data_accessed", "STRING"),
        bigquery.SchemaField("tables_queried", "STRING", mode="REPEATED"),
        bigquery.SchemaField("privacy_level", "STRING"),
        bigquery.SchemaField("k_anonymity_threshold", "INT64"),
        bigquery.SchemaField("differential_privacy_epsilon", "FLOAT64"),
        bigquery.SchemaField("hipaa_compliant", "BOOL"),
        bigquery.SchemaField("legal_basis", "STRING"),
        bigquery.SchemaField("purpose", "STRING"),
        bigquery.SchemaField("details", "JSON"),
        bigquery.SchemaField("client_ip", "STRING"),
        bigquery.SchemaField("session_id", "STRING"),
    ]
    
    try:
        table = bigquery.Table(FULL_TABLE_ID, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="event_timestamp"
        )
        table = client.create_table(table)
        print_result(True, f"Table created: {FULL_TABLE_ID}")
        return True
    except Exception as e:
        if "Already Exists" in str(e):
            print_result(True, "Table already exists")
            return True
        print_result(False, "Failed to create table", str(e))
        return False


# ============================================================================
# STEP 5: Test Insert
# ============================================================================

def test_insert(client):
    print_step(5, "Test Row Insert")
    
    test_row = {
        "log_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "hospital_id": HOSPITAL_ID,
        "action": "diagnostic_test",
        "action_category": "test",
        "event_timestamp": datetime.now().isoformat(),
        "logged_at": datetime.now().isoformat(),
        "user_id": "diagnostic_script",
        "patient_age_bucket": "60-65",
        "risk_score": 0.25,
        "risk_level": "LOW",
        "privacy_level": "k_anonymity_5",
        "k_anonymity_threshold": 5,
        "hipaa_compliant": True,
        "details": json.dumps({"test": True, "source": "diagnostic_script"})
    }
    
    print(f"\n  Test row:")
    for k, v in test_row.items():
        print(f"      {k}: {v}")
    
    try:
        errors = client.insert_rows_json(FULL_TABLE_ID, [test_row])
        
        if errors:
            print_result(False, "Insert returned errors")
            for error in errors:
                print(f"      Error: {error}")
            
            # Common error analysis
            error_str = str(errors)
            if "schema" in error_str.lower():
                print("\n  DIAGNOSIS: Schema mismatch")
                print("  FIX: Drop and recreate the table, or check column types")
            elif "permission" in error_str.lower():
                print("\n  DIAGNOSIS: Permission denied")
                print("  FIX: Grant roles/bigquery.dataEditor to your service account")
            
            return False
        else:
            print_result(True, "Row inserted successfully!")
            print(f"\n  Verify with:")
            print(f"      SELECT * FROM `{FULL_TABLE_ID}` WHERE log_id = '{test_row['log_id']}'")
            return True
            
    except Exception as e:
        print_result(False, "Insert threw exception", str(e))
        
        # Detailed error analysis
        error_str = str(e)
        if "404" in error_str or "Not found" in error_str:
            print("\n  DIAGNOSIS: Table not found")
            print("  FIX: Run this script again to create the table")
        elif "403" in error_str or "permission" in error_str.lower():
            print("\n  DIAGNOSIS: Permission denied")
            print("  FIX: Check IAM roles for your service account:")
            print("       - roles/bigquery.dataEditor")
            print("       - roles/bigquery.user")
        elif "schema" in error_str.lower():
            print("\n  DIAGNOSIS: Schema mismatch")
            print("  FIX: The table schema doesn't match. Options:")
            print("       1. Drop the table: bq rm -t {FULL_TABLE_ID}")
            print("       2. Run this script again to recreate")
        
        return False


# ============================================================================
# STEP 6: Verify Data
# ============================================================================

def verify_data(client):
    print_step(6, "Verify Data in Table")
    
    query = f"""
    SELECT 
        log_id,
        hospital_id,
        action,
        event_timestamp,
        user_id
    FROM `{FULL_TABLE_ID}`
    ORDER BY event_timestamp DESC
    LIMIT 5
    """
    
    try:
        results = client.query(query).result()
        rows = list(results)
        
        if rows:
            print_result(True, f"Found {len(rows)} recent rows")
            print("\n  Recent entries:")
            for row in rows:
                print(f"      {row.event_timestamp} | {row.action} | {row.user_id} | {row.log_id}")
        else:
            print_result(False, "Table is empty", "No rows found")
            print("\n  This means inserts are failing silently or the function isn't being called")
        
        return len(rows) > 0
        
    except Exception as e:
        print_result(False, "Query failed", str(e))
        return False


# ============================================================================
# STEP 7: Check audit_logging.py
# ============================================================================

def check_audit_module():
    print_step(7, "Check audit_logging.py Module")
    
    # Try to import the module
    try:
        # Add common paths
        possible_paths = [
            '.',
            './tools',
            './subagents/privacy_guardian/tools',
            '../tools',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                sys.path.insert(0, os.path.abspath(path))
        
        from audit_logging import log_prediction_access, _log_to_bigquery_audit
        print_result(True, "audit_logging module imported")
        
        # Check if the function has the right signature
        import inspect
        sig = inspect.signature(_log_to_bigquery_audit)
        print(f"      _log_to_bigquery_audit params: {list(sig.parameters.keys())[:5]}...")
        
        return True
        
    except ImportError as e:
        print_result(False, "Cannot import audit_logging", str(e))
        return False
    except Exception as e:
        print_result(False, "Error checking module", str(e))
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║           BIGQUERY AUDIT LOGGING DIAGNOSTIC                        ║
╠═══════════════════════════════════════════════════════════════════╣
║  Project:   {:<52} ║
║  Hospital:  {:<52} ║
║  Dataset:   {:<52} ║
║  Table:     {:<52} ║
╚═══════════════════════════════════════════════════════════════════╝
    """.format(PROJECT_ID, HOSPITAL_ID, AUDIT_DATASET, AUDIT_TABLE))
    
    # Step 1: Check BigQuery client
    client = check_bigquery_client()
    if not client:
        print("\n\n❌ STOPPED: Cannot proceed without BigQuery client")
        return 1
    
    # Step 2: Check dataset
    if not check_dataset(client):
        print("\n\n❌ STOPPED: Cannot proceed without dataset")
        return 1
    
    # Step 3: Check table
    schema = check_table(client)
    
    # Step 4: Create table if missing
    if not schema:
        if not create_table(client):
            print("\n\n❌ STOPPED: Cannot create table")
            return 1
    
    # Step 5: Test insert
    insert_ok = test_insert(client)
    
    # Step 6: Verify data
    verify_data(client)
    
    # Step 7: Check module
    check_audit_module()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    if insert_ok:
        print("""
  ✓ BigQuery connection: WORKING
  ✓ Table insert: WORKING
  
  If your agent still isn't logging to BigQuery, the issue is likely:
  
  1. The audit functions aren't being CALLED by the agent
     → Check if log_prediction_access() is actually invoked
     → Add print statements to verify
  
  2. Errors are being swallowed silently
     → Check the except blocks in audit_logging.py
     → Make sure errors are logged, not just caught
  
  3. The agent is using a DIFFERENT audit_logging.py
     → Check your import paths
     → Verify which file is actually being loaded
        """)
    else:
        print("""
  ✗ BigQuery insert: FAILED
  
  Check the error messages above for specific fixes.
  Common issues:
  
  1. Permission denied → Add roles/bigquery.dataEditor
  2. Schema mismatch → Drop and recreate table
  3. Table not found → Run create table step
        """)
    
    print(f"""
  To manually check BigQuery:
  
    bq query --use_legacy_sql=false \\
      'SELECT * FROM `{FULL_TABLE_ID}` ORDER BY event_timestamp DESC LIMIT 5'
    """)
    
    return 0 if insert_ok else 1


if __name__ == "__main__":
    sys.exit(main())