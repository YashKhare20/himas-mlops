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
# from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping  # DP disabled
from himas_model_pipeline.task import (
    load_model, get_shared_hyperparameters,
    get_config_value, set_seed, load_hyperparameters
)

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

    # Get feature dimensions
    num_features = get_config_value('tool.himas.data.numerical-features', [])
    cat_features = get_config_value('tool.himas.data.categorical-features', [])
    input_dim = len(num_features) + len(cat_features)
    logger.info(
        f"  Input dimension: {input_dim} ({len(num_features)} numerical + {len(cat_features)} categorical)")

    # Load shared hyperparameters
    hyperparameters = get_shared_hyperparameters(context)
    hp_path = get_config_value('tool.flwr.app.config.shared-hyperparameters')

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

    # Initialize FedAvg strategy
    logger.info("-"*70)
    logger.info("Initializing FedAvg strategy")
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=3
    )

    # Apply Differential Privacy if enabled
    # DP code commented out - not working properly
    # enable_dp = context.run_config.get(
    #     "enable-differential-privacy",
    #     get_config_value('tool.flwr.app.config.enable-differential-privacy', False)
    # )
    # 
    # # Initialize DP parameters (for metadata tracking)
    # dp_noise_multiplier = None
    # dp_clipping_norm = None
    # dp_num_sampled_clients = None
    # 
    # if enable_dp:
    #     dp_noise_multiplier = context.run_config.get(
    #         "dp-noise-multiplier",
    #         get_config_value('tool.flwr.app.config.dp-noise-multiplier', 1.0)
    #     )
    #     dp_clipping_norm = context.run_config.get(
    #         "dp-clipping-norm",
    #         get_config_value('tool.flwr.app.config.dp-clipping-norm', 2.0)
    #     )
    #     dp_num_sampled_clients = context.run_config.get(
    #         "dp-num-sampled-clients",
    #         get_config_value('tool.flwr.app.config.dp-num-sampled-clients', 3)
    #     )
    #     
    #     strategy = DifferentialPrivacyServerSideFixedClipping(
    #         strategy,
    #         noise_multiplier=dp_noise_multiplier,
    #         clipping_norm=dp_clipping_norm,
    #         num_sampled_clients=dp_num_sampled_clients,
    #     )
    #     logger.info(f"Differential Privacy enabled:")
    #     logger.info(f"  Noise multiplier: {dp_noise_multiplier}")
    #     logger.info(f"  Clipping norm: {dp_clipping_norm}")
    #     logger.info(f"  Sampled clients per round: {dp_num_sampled_clients}")
    # else:
    #     logger.info("Differential Privacy disabled")
    
    logger.info("Differential Privacy disabled")

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
    model.set_weights(result.arrays.to_numpy_ndarrays())
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
        'dataset_id': dataset_id,
        'differential_privacy': {
            'enabled': False,
            # 'noise_multiplier': dp_noise_multiplier,
            # 'clipping_norm': dp_clipping_norm,
            # 'num_sampled_clients': dp_num_sampled_clients
        }
    }

    metadata_path = models_save_dir / f"model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata saved to {metadata_path}")
    logger.info("="*70)
    logger.info("SERVER TASK COMPLETED SUCCESSFULLY")
    logger.info("="*70)
