"""
Test DAG for Email Functionality and Airflow 3.1 Context
Tests SMTP configuration and context-based arguments.
"""
from datetime import datetime
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.email import send_email
import logging, os

logger = logging.getLogger(__name__)


def test_context_variables(**context):
    """
    Test and log all Airflow 3.1 context variables.
    This helps debug what's available in the context.
    """
    logger.info("=" * 60)
    logger.info("AIRFLOW 3.1 CONTEXT VARIABLES TEST")
    logger.info("=" * 60)

    # Test all documented context variables
    context_vars = {
        # Date/Time variables
        'logical_date': context.get('logical_date'),
        'ds': context.get('ds'),
        'ds_nodash': context.get('ds_nodash'),
        'ts': context.get('ts'),
        'ts_nodash': context.get('ts_nodash'),
        'start_date': context.get('start_date'),

        # DAG/Task info
        'dag': context.get('dag'),
        'dag_run': context.get('dag_run'),
        'task': context.get('task'),
        'task_instance': context.get('task_instance'),
        'ti': context.get('ti'),
        'run_id': context.get('run_id'),

        # Other useful variables
        'params': context.get('params'),
        'test_mode': context.get('test_mode'),
        'try_number': context.get('try_number'),
        'task_instance_key_str': context.get('task_instance_key_str'),
    }

    # Log each variable
    for key, value in context_vars.items():
        if value is not None:
            # Special handling for complex objects
            if key in ['dag', 'dag_run', 'task', 'task_instance', 'ti']:
                logger.info(f"{key}: {type(value).__name__}")
                if hasattr(value, 'dag_id'):
                    logger.info(f"  └─ dag_id: {value.dag_id}")
                if hasattr(value, 'task_id'):
                    logger.info(f"  └─ task_id: {value.task_id}")
            else:
                logger.info(f"{key}: {value}")
        else:
            logger.info(f"{key}: NOT AVAILABLE")

    logger.info("=" * 60)

    # Test accessing DAG run info
    dag_run = context.get('dag_run')
    if dag_run:
        logger.info("DAG RUN INFO:")
        logger.info(f"  dag_id: {dag_run.dag_id}")
        logger.info(f"  run_id: {dag_run.run_id}")
        logger.info(f"  Type: {type(dag_run).__name__}")

    # Test accessing DAG info
    dag = context.get('dag')
    if dag:
        logger.info("DAG INFO:")
        logger.info(f"  dag_id: {dag.dag_id}")
        logger.info(f"  task_count: {len(dag.task_ids)}")
        logger.info(f"  task_ids: {dag.task_ids}")

    logger.info("=" * 60)
    return "Context test completed"


def test_smtp_config(**context):
    """Test SMTP configuration and send email with context info"""
    import os

    # Log SMTP configuration (without password)
    logger.info("=" * 60)
    logger.info("SMTP CONFIGURATION DEBUG")
    logger.info("=" * 60)
    logger.info(
        f"SMTP_HOST: {os.getenv('AIRFLOW__SMTP__SMTP_HOST', 'NOT SET')}")
    logger.info(
        f"SMTP_PORT: {os.getenv('AIRFLOW__SMTP__SMTP_PORT', 'NOT SET')}")
    logger.info(
        f"SMTP_USER: {os.getenv('AIRFLOW__SMTP__SMTP_USER', 'NOT SET')}")
    logger.info(
        f"SMTP_PASSWORD: {'SET' if os.getenv('AIRFLOW__SMTP__SMTP_PASSWORD') else 'NOT SET'}")
    logger.info(
        f"SMTP_STARTTLS: {os.getenv('AIRFLOW__SMTP__SMTP_STARTTLS', 'NOT SET')}")
    logger.info("=" * 60)

    # Get context info
    dag_id = context.get('dag').dag_id if context.get('dag') else 'N/A'
    task_id = context.get('task').task_id if context.get('task') else 'N/A'
    logical_date = context.get('logical_date')
    ds = context.get('ds', 'N/A')
    run_id = context.get('run_id', 'N/A')

    logical_date_str = logical_date.strftime(
        '%Y-%m-%d %H:%M:%S') if logical_date else ds

    # Try sending email with context info
    try:
        send_email(
            to=os.getenv('ALERT_EMAILS'),
            subject=f'✅ Test Email from Airflow - {dag_id}',
            html_content=f'''
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; }}
                    .success {{ background-color: #d4edda; padding: 15px; border-radius: 5px; }}
                    .info {{ background-color: #d1ecf1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                    h1 {{ color: #28a745; }}
                    .detail {{ margin: 5px 0; }}
                </style>
            </head>
            <body>
                <div class="success">
                    <h1>✅ SMTP Configuration Working!</h1>
                    <p>Your Airflow email configuration is successful.</p>
                </div>

                <div class="info">
                    <h2>Context Information (Airflow 3.1)</h2>
                    <div class="detail"><strong>DAG ID:</strong> {dag_id}</div>
                    <div class="detail"><strong>Task ID:</strong> {task_id}</div>
                    <div class="detail"><strong>Run ID:</strong> {run_id}</div>
                    <div class="detail"><strong>Logical Date:</strong> {logical_date_str}</div>
                    <div class="detail"><strong>DS:</strong> {ds}</div>
                </div>

                <div class="info">
                    <h2>✓ Verified:</h2>
                    <ul>
                        <li>Gmail SMTP configured</li>
                        <li>Authentication successful</li>
                        <li>Email delivery confirmed</li>
                        <li>Context variables accessible</li>
                    </ul>
                </div>
            </body>
            </html>
            '''
        )
        logger.info("✅ Test email sent successfully!")
        return "Email sent successfully"
    except Exception as e:
        logger.error(f"❌ Failed to send email: {str(e)}")
        logger.exception("Full error traceback:")
        raise


def test_success_callback(context):
    """
    Test success callback with context (simulates DAG-level callback).
    This function tests what's available in on_success_callback.
    """
    logger.info("=" * 60)
    logger.info("SUCCESS CALLBACK CONTEXT TEST")
    logger.info("=" * 60)

    # Get available context
    dag = context.get('dag')
    dag_run = context.get('dag_run')
    logical_date = context.get('logical_date')
    start_date = context.get('start_date')

    logger.info(f"DAG: {dag.dag_id if dag else 'N/A'}")
    logger.info(f"DAG Run: {dag_run.dag_id if dag_run else 'N/A'}")
    logger.info(f"Logical Date: {logical_date}")
    logger.info(f"Start Date: {start_date}")

    if dag:
        logger.info(f"Task Count: {len(dag.task_ids)}")
        logger.info(f"Tasks: {dag.task_ids}")

    # Try sending success email
    try:
        dag_id = dag.dag_id if dag else 'test_email'

        send_email(
            to=os.getenv('ALERT_EMAILS'),
            subject=f'✅ Success Callback Test - {dag_id}',
            html_content=f'''
            <html>
            <body>
                <h1 style="color: #28a745;">✅ Success Callback Working!</h1>
                <p>The on_success_callback is functioning correctly.</p>
                <h2>Context Available:</h2>
                <ul>
                    <li>DAG ID: {dag_id}</li>
                    <li>Logical Date: {logical_date}</li>
                    <li>Start Date: {start_date}</li>
                </ul>
            </body>
            </html>
            '''
        )
        logger.info("✅ Success callback email sent!")
    except Exception as e:
        logger.error(f"❌ Success callback email failed: {str(e)}")

    logger.info("=" * 60)


def fail_task():
    """Task that intentionally fails to test failure email"""
    raise Exception(
        "🔴 This is an intentional test failure to verify email alerts")


with DAG(
    dag_id='test_email',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        'owner': 'airflow',
        'email': os.getenv('ALERT_EMAILS'),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 0,  # No retries for testing
    },
    on_success_callback=test_success_callback,  # Test DAG-level success callback
    tags=['test', 'email', 'context'],
    doc_md="""
    # Email & Context Test DAG

    This DAG tests:
    1. Airflow 3.1 context variables
    2. SMTP email configuration
    3. Task-level emails (success/failure)
    4. DAG-level success callback

    ## Tasks:
    - `test_context`: Logs all available context variables
    - `test_smtp`: Sends test email with context info
    - `test_failure`: Fails to test failure email (uncomment to use)
    """
) as dag:

    # Task 1: Test context variables
    test_context = PythonOperator(
        task_id='test_context_variables',
        python_callable=test_context_variables,
        doc_md="Tests and logs all Airflow 3.1 context variables"
    )

    # Task 2: Test SMTP configuration
    test_smtp = PythonOperator(
        task_id='test_smtp_config',
        python_callable=test_smtp_config,
        doc_md="Sends test email with context information"
    )

    # Task 3: Test failure email (commented by default)
    # Uncomment this task to test failure emails
    test_failure = PythonOperator(
        task_id='test_failure_email',
        python_callable=fail_task,
        doc_md="Intentionally fails to test failure email alerts"
    )

    # Set task dependencies
    test_context >> test_smtp
    test_smtp >> test_failure  # Uncomment if testing failure
