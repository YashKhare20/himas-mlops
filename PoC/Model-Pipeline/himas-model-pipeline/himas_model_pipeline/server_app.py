"""
HIMAS ServerApp - Federated Server for ICU Mortality Prediction
================================================================

Coordinates federated learning across three hospitals:
- Initializes global model with shared hyperparameters
- Orchestrates training rounds using FedAvg strategy
- Aggregates model weights from hospital clients
- Saves final model with hyperparameter-specific naming
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from himas_model_pipeline.task import (
    load_model, get_shared_hyperparameters,
    get_config_value, set_seed, load_hyperparameters
)

import os
import mlflow
import mlflow.keras

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
_MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT_NAME", "himas-federated")
mlflow.set_tracking_uri(_MLFLOW_URI)
mlflow.set_experiment(_MLFLOW_EXP)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for federated learning server.

    Orchestrates the complete federated learning workflow:
    1. Load configuration and hyperparameters
    2. Initialize global model
    3. Execute federated training rounds
    4. Save final aggregated model

    Args:
        grid: Flower Grid for client communication
        context: Flower Context with runtime configuration
    """
    logger.info("="*70)
    logger.info("HIMAS FEDERATED LEARNING - ICU MORTALITY PREDICTION")
    logger.info("="*70)

    # --- MLflow (server-run) setup: non-intrusive additions ---
    import os
    import mlflow
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Respect env if provided by docker-compose
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "himas-federated")
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=f"server_{run_ts}")
        mlflow.set_tags({"role": "server", "phase": "federated-training"})
    except Exception as e:
        logger.warning(f"MLflow setup warning: {e}")
    # ----------------------------------------------------------

    # Configure reproducibility
    seed = context.run_config.get(
        "random-seed", get_config_value('tool.flwr.app.config.random-seed', 42))
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Load runtime configuration
    project_id = get_config_value('tool.himas.data.project-id')
    dataset_id = get_config_value('tool.himas.data.dataset-id')
    num_rounds = context.run_config.get("num-server-rounds", 15)
    fraction_train = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate = context.run_config.get("fraction-evaluate", 1.0)

    logger.info(f"Configuration:")
    logger.info(f"  Project: {project_id}")
    logger.info(f"  Dataset: {dataset_id}")
    logger.info(f"  Federated rounds: {num_rounds}")
    logger.info(f"  Train fraction: {fraction_train}")
    logger.info(f"  Evaluate fraction: {fraction_evaluate}")

    # --- MLflow: log config params/tags (added) ---
    try:
        mlflow.log_params({
            "num_server_rounds": num_rounds,
            "fraction_train": fraction_train,
            "fraction_evaluate": fraction_evaluate,
            "random_seed": seed,
        })
        mlflow.set_tags({
            "project_id": str(project_id),
            "dataset_id": str(dataset_id),
            "strategy": "FedAvg",
        })
    except Exception as e:
        logger.warning(f"MLflow param/tag logging warning: {e}")
    # ----------------------------------------------

    # Get feature dimensions
    num_features = get_config_value('tool.himas.data.numerical-features', [])
    cat_features = get_config_value('tool.himas.data.categorical-features', [])
    input_dim = len(num_features) + len(cat_features)
    logger.info(
        f"  Input dimension: {input_dim} ({len(num_features)} numerical + {len(cat_features)} categorical)")

    # --- MLflow: log input_dim (added) ---
    try:
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("num_numerical_features", len(num_features))
        mlflow.log_param("num_categorical_features", len(cat_features))
    except Exception as e:
        logger.warning(f"MLflow input_dim logging warning: {e}")
    # -------------------------------------

    # Load shared hyperparameters
    hyperparameters = get_shared_hyperparameters(context)
    hp_path = get_config_value('tool.flwr.app.config.shared-hyperparameters')

    # --- MLflow: log hyperparameters summary and attach JSON if available (added) ---
    try:
        if isinstance(hyperparameters, dict):
            # log compactly; avoid huge nested dumps
            compact_hp = {k: hyperparameters[k] for k in sorted(hyperparameters.keys())}
            mlflow.log_dict(compact_hp, "hyperparameters_used.json")
        if hp_path and Path(hp_path).exists():
            mlflow.log_artifact(hp_path, artifact_path="hyperparameters")
    except Exception as e:
        logger.warning(f"MLflow hyperparameters logging warning: {e}")
    # --------------------------------------------------------------------------------

    # Extract hospital identifier from hyperparameters path for organized model saving
    if hp_path:
        # e.g., 'hospital_a_best_hyperparameters'
        hp_filename = Path(hp_path).stem
        # Extract hospital name (e.g., 'hospital_a')
        hosp_identifier = hp_filename.split(
            '_best_')[0] if '_best_' in hp_filename else 'default'
    else:
        hosp_identifier = 'default'

    logger.info("Building global model")
    model = load_model(input_dim, hyperparameters, seed)

    # Log model statistics
    total_params = model.count_params()
    logger.info(f"Model parameters: {total_params:,}")

    # --- MLflow: log model param count (added) ---
    try:
        mlflow.log_param("total_parameters", int(total_params))
    except Exception as e:
        logger.warning(f"MLflow total_parameters logging warning: {e}")
    # ---------------------------------------------

    # Initialize FedAvg strategy
    logger.info("-"*70)
    logger.info("Initializing FedAvg strategy")
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2 #changed from 3 for testing with less clients
    )

    # Execute federated learning
    logger.info("="*70)
    logger.info(
        f"Starting federated learning across 3 hospitals for {num_rounds} rounds")
    logger.info("="*70)

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.get_weights()),
        num_rounds=num_rounds
    )

    # Save final model with organized directory structure
    logger.info("="*70)
    logger.info("FEDERATED LEARNING COMPLETED")
    logger.info("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory structure: models/hyper-{hospital}/
    models_base_dir = Path(get_config_value(
        'tool.himas.paths.models-dir', 'models'))
    models_save_dir = models_base_dir / f"hyper-{hosp_identifier}"
    models_save_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"himas_federated_mortality_model_{timestamp}.keras"
    model_path = models_save_dir / model_filename

    logger.info(f"Saving final model:")
    logger.info(f"  Directory: {models_save_dir}")
    logger.info(f"  Filename: {model_filename}")
    logger.info(f"  Full path: {model_path}")

    # Save model weights
    #model.set_weights(result.arrays.to_numpy_ndarrays())

    #Added to avoid issues with saving in some environments
    # Save model weights - with safety check
    ndarrays = result.arrays.to_numpy_ndarrays()
    logger.info(f"Received {len(ndarrays)} weight arrays from federated training")
    
    if len(ndarrays) == 0:
        raise RuntimeError(
            "❌ Training failed - no weights returned from clients!\n"
            "Check client logs above for errors during training."
        )
    
    logger.info("✅ Valid weights received, setting model weights")
    model.set_weights(ndarrays)


    model.save(str(model_path))

    # Save model metadata
    metadata = {
        'timestamp': timestamp,
        'hyperparameters_source': hosp_identifier,
        'hyperparameters_path': hp_path,
        'hyperparameters': hyperparameters,
        'model_path': str(model_path),
        'num_rounds': num_rounds,
        'total_parameters': total_params,
        'random_seed': seed,
        'project_id': project_id,
        'dataset_id': dataset_id
    }

    metadata_path = models_save_dir / f"model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata saved to {metadata_path}")
    logger.info("="*70)
    logger.info("SERVER TASK COMPLETED SUCCESSFULLY")
    logger.info("="*70)

    # --- MLflow: log artifacts and finalize run (added) ---
    try:
        # Log where/what we saved
        mlflow.log_param("saved_model_dir", str(models_save_dir))
        mlflow.log_param("saved_model_path", str(model_path))
        # Attach artifacts
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="model")
        if metadata_path.exists():
            mlflow.log_artifact(str(metadata_path), artifact_path="model")
    except Exception as e:
        logger.warning(f"MLflow artifact logging warning: {e}")
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass
    # -------------------------------------------------------
