"""himas-model-pipeline: A Flower / TensorFlow ServerApp for federated ICU mortality prediction."""

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, DifferentialPrivacyServerSideFixedClipping
from himas_model_pipeline.task import load_model, set_random_seed
from pathlib import Path

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the ServerApp.

    Coordinates federated learning across 3 hospitals for ICU mortality prediction.
    """

    # Set random seed from config for reproducibility
    random_seed = context.run_config.get("random-seed", 42)
    set_random_seed(random_seed)

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)

    # Read DP config
    enable_dp = context.run_config.get("enable-differential-privacy", False)
    dp_noise_multiplier = context.run_config.get("dp-noise-multiplier", 1.0)
    dp_clipping_norm = context.run_config.get("dp-clipping-norm", 1.0)
    dp_num_sampled_clients = context.run_config.get("dp-num-sampled-clients", 3)

    print("\n" + "="*60)
    print("HIMAS Federated Learning - ICU Mortality Prediction")
    print("="*60)
    print(f"Number of federated rounds: {num_rounds}")
    print(f"Fraction of clients for training: {fraction_train}")
    print(f"Fraction of clients for evaluation: {fraction_evaluate}")
    print("="*60 + "\n")

    # Initialize global model with proper input dimension
    # Note: Feature dimension will be set after first data load
    # For now, use a placeholder and the model will be properly initialized
    # after the first round when clients report their data
    input_dim = 23  # 15 numerical + 8 categorical features (approximate)
    model = load_model(input_dim)
    arrays = ArrayRecord(model.get_weights())

    # Initialize FedAvg strategy with healthcare-specific configuration
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=2,  # Minimum 2 hospitals for training
        min_evaluate_nodes=2,  # Minimum 2 hospitals for evaluation
        min_available_nodes=3,  # All 3 hospitals should be available
    )

    # Wrap with DP if enabled
    if enable_dp:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=dp_noise_multiplier,
            clipping_norm=dp_clipping_norm,
            num_sampled_clients=dp_num_sampled_clients,
        )
        print(f"Differential Privacy enabled (noise={dp_noise_multiplier})")

    # Start federated learning
    print("Starting federated learning across hospitals...")
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model
    print("\n" + "="*60)
    print("Federated learning completed!")
    print("="*60)

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "himas_federated_mortality_model.keras"

    print(f"\nSaving final model to: {model_path}")
    ndarrays = result.arrays.to_numpy_ndarrays()
    model.set_weights(ndarrays)
    model.save(str(model_path))

    print("\nModel saved successfully")
    print("="*60 + "\n")
