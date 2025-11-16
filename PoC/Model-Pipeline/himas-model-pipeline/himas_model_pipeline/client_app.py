"""
HIMAS ClientApp - Federated Client for ICU Mortality Prediction
================================================================

Implements Flower ClientApp for federated learning with:
- Hospital-specific data loading and preprocessing
- Local model training with class weight balancing
- Validation monitoring with early stopping
- Round-based metrics tracking with JSON export

Each hospital client trains independently on its local data while participating
in federated weight aggregation coordinated by the ServerApp.
"""

import logging
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from himas_model_pipeline.task import (
    load_data_from_bigquery, load_model, get_feature_dim,
    get_shared_hyperparameters, set_seed, get_config_value
)
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flower ClientApp
app = ClientApp()

# Output directory for metrics
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
METRICS_DIR = Path("training_results")
TRAINING_METRICS_FILE = METRICS_DIR / f"training_metrics_{timestamp}.json"
EVALUATION_METRICS_FILE = METRICS_DIR / f"evaluation_metrics_{timestamp}.json"


def load_metrics_file(filepath: Path) -> dict:
    """Load existing metrics file or return empty dict."""
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def save_metrics_file(data: dict, filepath: Path):
    """Save metrics dictionary to JSON file."""
    METRICS_DIR.mkdir(exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Metrics saved to {filepath}")


def get_round_number(context: Context) -> int:
    """Extract round number from context."""
    round_num = context.run_config.get("server-round", None)

    if round_num is not None:
        return int(round_num)

    # Fallback: infer from existing training metrics
    existing_metrics = load_metrics_file(TRAINING_METRICS_FILE)
    if existing_metrics:
        # Extract round numbers from keys like "round_1_hospital_a_partition_0"
        round_numbers = [int(k.split('_')[1])
                         for k in existing_metrics.keys() if k.startswith('round_')]
        if round_numbers:
            return max(round_numbers) + 1

    return 1  # First round


def create_metric_key(round_num: int, hospital_name: str, partition_id: int) -> str:
    """Create standardized metric key."""
    return f"round_{round_num}_{hospital_name}_partition_{partition_id}"


@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Train local model on hospital-specific data.

    Implements federated learning training phase where each hospital:
    1. Receives global model weights from server
    2. Trains on local data with validation monitoring
    3. Returns updated weights and performance metrics to server

    Args:
        msg: Message containing global model weights from server
        context: Flower context with configuration and node information

    Returns:
        Message containing updated model weights and training metrics

    Note:
        Uses class weights to handle mortality class imbalance.
        Employs early stopping and learning rate reduction for optimal convergence.
    """
    logger.info("="*70)
    logger.info("TRAINING TASK STARTED")
    logger.info("="*70)

    # Get round number
    round_num = get_round_number(context)

    # Configure reproducibility
    seed = context.run_config.get(
        "random-seed", get_config_value('tool.flwr.app.config.random-seed', 42))
    set_seed(seed)

    # Get hospital assignment
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    hospital_names = get_config_value(
        'tool.himas.data.hospital-names', ['hospital_a', 'hospital_b', 'hospital_c'])
    hospital_name = hospital_names[partition_id]

    logger.info(
        f"Round {round_num} | Hospital: {hospital_name} (Partition {partition_id}/{num_partitions})")

    # Load hospital data
    x_train, y_train, x_val, y_val = load_data_from_bigquery(
        partition_id, num_partitions)

    # Build model with shared architecture
    input_dim = get_feature_dim()
    hyperparameters = get_shared_hyperparameters(context)
    model = load_model(input_dim, hyperparameters, seed)

    # Initialize with global weights
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Training configuration
    epochs = context.run_config.get("local-epochs", 5)
    batch_size = context.run_config.get("batch-size", 64)
    verbose = context.run_config.get("verbose", 1)

    # Compute class weights for mortality imbalance
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")

    # Configure training callbacks
    callbacks = []

    if context.run_config.get("use-early-stopping", True):
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=context.run_config.get("early-stopping-patience", 3),
            mode='max',
            restore_best_weights=True,
            verbose=1
        ))

    if context.run_config.get("use-reduce-lr", True):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=context.run_config.get("reduce-lr-factor", 0.5),
            patience=context.run_config.get("reduce-lr-patience", 2),
            min_lr=context.run_config.get("reduce-lr-min", 1e-7),
            verbose=1
        ))

    logger.info(
        f"Training: {len(x_train):,} samples, max {epochs} epochs, batch_size={batch_size}")

    # Train with validation monitoring
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose
    )

    # Extract metrics
    actual_epochs = len(history.history['loss'])

    metrics = {
        "hospital": hospital_name,
        "partition_id": partition_id,
        "n_train": len(x_train),
        "n_val": len(x_val),
        "epochs": actual_epochs,
        "train_loss": float(history.history['loss'][-1]),
        "train_acc": float(history.history['accuracy'][-1]),
        "train_auc": float(history.history['auc'][-1]),
        "train_precision": float(history.history['precision'][-1]),
        "train_recall": float(history.history['recall'][-1]),
        "val_loss": float(history.history['val_loss'][-1]),
        "val_acc": float(history.history['val_accuracy'][-1]),
        "val_auc": float(history.history['val_auc'][-1]),
        "val_precision": float(history.history['val_precision'][-1]),
        "val_recall": float(history.history['val_recall'][-1])
    }

    # Calculate F1 scores
    train_f1 = 2 * metrics['train_precision'] * metrics['train_recall'] / (
        metrics['train_precision'] + metrics['train_recall']) if (metrics['train_precision'] + metrics['train_recall']) > 0 else 0.0
    val_f1 = 2 * metrics['val_precision'] * metrics['val_recall'] / (
        metrics['val_precision'] + metrics['val_recall']) if (metrics['val_precision'] + metrics['val_recall']) > 0 else 0.0

    metrics['train_f1'] = float(train_f1)
    metrics['val_f1'] = float(val_f1)

    # Create round record
    round_record = {
        'timestamp': datetime.now().isoformat(),
        'hospital': hospital_name,
        'partition_id': partition_id,
        'round': round_num,
        'epochs_run': actual_epochs,
        'final_metrics': metrics,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }

    # Create key with hospital and partition info
    metric_key = create_metric_key(round_num, hospital_name, partition_id)

    # Load existing metrics and add this round
    all_training_metrics = load_metrics_file(TRAINING_METRICS_FILE)
    all_training_metrics[metric_key] = round_record
    save_metrics_file(all_training_metrics, TRAINING_METRICS_FILE)

    # Log concise summary
    logger.info("-"*70)
    logger.info(
        f"Training Complete (Round {round_num}, {actual_epochs} epochs):")
    logger.info(f"  Train: Loss={metrics['train_loss']:.4f}, AUC={metrics['train_auc']:.4f}, "
                f"Precision={metrics['train_precision']:.3f}, Recall={metrics['train_recall']:.3f}")
    logger.info(f"  Val:   Loss={metrics['val_loss']:.4f}, AUC={metrics['val_auc']:.4f}, "
                f"Precision={metrics['val_precision']:.3f}, Recall={metrics['val_recall']:.3f}")
    logger.info("-"*70)

    # Return concise metrics to server
    return_metrics = {
        "num-examples": len(x_train),
        "train_loss": metrics['train_loss'],
        "train_auc": metrics['train_auc'],
        "val_auc": metrics['val_auc']
    }

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.get_weights()),
            "metrics": MetricRecord(return_metrics)
        }),
        reply_to=msg
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Evaluate global model on local validation data.

    Args:
        msg: Message containing global model weights from server
        context: Flower context with configuration

    Returns:
        Message containing evaluation metrics
    """

    logger.info("="*70)
    logger.info("EVALUATION TASK STARTED")
    logger.info("="*70)

    # Get round number
    round_num = get_round_number(context)

    # Configure reproducibility
    seed = context.run_config.get(
        "random-seed", get_config_value('tool.flwr.app.config.random-seed', 42))
    set_seed(seed)

    # Get hospital assignment
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    hospital_names = get_config_value(
        'tool.himas.data.hospital-names', ['hospital_a', 'hospital_b', 'hospital_c'])
    hospital_name = hospital_names[partition_id]

    logger.info(
        f"Round {round_num} | Hospital: {hospital_name} (Partition {partition_id})")

    # Load validation data
    _, _, x_val, y_val = load_data_from_bigquery(partition_id, num_partitions)

    # Build model and load global weights
    input_dim = get_feature_dim()
    hyperparameters = get_shared_hyperparameters(context)
    model = load_model(input_dim, hyperparameters, seed)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Evaluate
    logger.info(f"Evaluating on {len(x_val):,} validation samples")
    results = model.evaluate(x_val, y_val, verbose=0)

    # Extract metrics
    metrics = {
        "hospital": hospital_name,
        "partition_id": partition_id,
        "n_samples": len(x_val),
        "eval_loss": float(results[0]),
        "eval_acc": float(results[1]),
        "eval_auc": float(results[2]),
        "eval_precision": float(results[3]),
        "eval_recall": float(results[4])
    }

    # Calculate F1
    eval_f1 = 2 * metrics['eval_precision'] * metrics['eval_recall'] / (
        metrics['eval_precision'] + metrics['eval_recall']) if (metrics['eval_precision'] + metrics['eval_recall']) > 0 else 0.0
    metrics['eval_f1'] = float(eval_f1)

    # Create round record
    round_record = {
        'timestamp': datetime.now().isoformat(),
        'hospital': hospital_name,
        'partition_id': partition_id,
        'round': round_num,
        'metrics': metrics
    }

    # Create key with hospital and partition info
    metric_key = create_metric_key(round_num, hospital_name, partition_id)

    # Load existing metrics and add this round
    all_evaluation_metrics = load_metrics_file(EVALUATION_METRICS_FILE)
    all_evaluation_metrics[metric_key] = round_record
    save_metrics_file(all_evaluation_metrics, EVALUATION_METRICS_FILE)

    # Log concise summary
    logger.info("-"*70)
    logger.info(f"Evaluation Results (Round {round_num}):")
    logger.info(f"  Loss={metrics['eval_loss']:.4f}, AUC={metrics['eval_auc']:.4f}, "
                f"Precision={metrics['eval_precision']:.3f}, Recall={metrics['eval_recall']:.3f}, F1={metrics['eval_f1']:.3f}")
    logger.info("-"*70)

    # Return concise metrics to server
    return_metrics = {
        "num-examples": len(x_val),
        "eval_loss": metrics['eval_loss'],
        "eval_auc": metrics['eval_auc'],
        "eval_precision": metrics['eval_precision'],
        "eval_recall": metrics['eval_recall']
    }

    return Message(
        content=RecordDict({"metrics": MetricRecord(return_metrics)}),
        reply_to=msg
    )
