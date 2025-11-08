"""himas-model-pipeline: A Flower / TensorFlow ServerApp for federated ICU mortality prediction."""

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from himas_model_pipeline.task import load_model
from pathlib import Path

# --- MLflow (minimal additions) ---
import os
import mlflow
import mlflow.keras


# Configure MLflow from env or local defaults
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
_MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT_NAME", "himas-federated")
mlflow.set_tracking_uri(_MLFLOW_URI)
mlflow.set_experiment(_MLFLOW_EXP)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the ServerApp.

    Coordinates federated learning across 3 hospitals for ICU mortality prediction.
    """
    # Start a top-level MLflow run for the full FL session
    with mlflow.start_run(run_name="server-session"):
        # Read run config
        num_rounds: int = context.run_config["num-server-rounds"]
        fraction_train: float = context.run_config.get("fraction-train", 1.0)
        fraction_evaluate: float = context.run_config.get("fraction-evaluate", 1.0)

        print("\n" + "="*60)
        print("HIMAS Federated Learning - ICU Mortality Prediction")
        print("="*60)
        print(f"Number of federated rounds: {num_rounds}")
        print(f"Fraction of clients for training: {fraction_train}")
        print(f"Fraction of clients for evaluation: {fraction_evaluate}")
        print("="*60 + "\n")

        # Log run-level parameters
        mlflow.set_tags({"role": "server", "phase": "federated-session"})
        mlflow.log_params({
            "num_server_rounds": num_rounds,
            "fraction_train": fraction_train,
            "fraction_evaluate": fraction_evaluate,
        })

        # Initialize global model with proper input dimension
        input_dim = 23  # 15 numerical + 8 categorical features (approximate)
        model = load_model(input_dim)
        arrays = ArrayRecord(model.get_weights())

        # Enable Keras autologging (logs training/eval metrics from clients if emitted)
        mlflow.keras.autolog(log_models=False)

        # Initialize FedAvg strategy with healthcare-specific configuration
        strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=2,    # Minimum 2 hospitals for training
            min_evaluate_nodes=2, # Minimum 2 hospitals for evaluation
            min_available_nodes=3,# All 3 hospitals should be available
        )

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

        # Track artifacts/model in MLflow (kept simple)
        mlflow.log_artifact(str(model_path), artifact_path="final_model_artifacts")
        # Also log as an MLflow model (optional, convenient for loading)
        mlflow.keras.log_model(model, artifact_path="final_model")

        print("\nModel saved successfully")
        print("="*60 + "\n")
