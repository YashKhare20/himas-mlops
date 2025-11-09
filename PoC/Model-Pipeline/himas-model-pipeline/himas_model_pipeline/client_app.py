"""himas-model-pipeline: A Flower / TensorFlow app for federated ICU mortality prediction."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from himas_model_pipeline.task import load_data_from_bigquery, load_model, get_feature_dim, set_random_seed
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local hospital data."""

    # Set random seed from config for reproducibility
    random_seed = context.run_config.get("random-seed", 42)
    set_random_seed(random_seed)

    # Load the data for this hospital
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    x_train, y_train, _, _ = load_data_from_bigquery(
        partition_id, num_partitions)

    # Get feature dimension and load model
    input_dim = get_feature_dim()
    model = load_model(input_dim)

    # Initialize model with received weights
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(ndarrays)

    # Read training config
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose", 0)

    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Train the model on local hospital data
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=verbose,
        shuffle=True,  # Shuffle for better training, but with fixed seed for reproducibility
    )

    # Extract metrics
    train_loss = history.history["loss"][-1] if "loss" in history.history else None
    train_acc = history.history["accuracy"][-1] if "accuracy" in history.history else None
    train_auc = history.history["auc"][-1] if "auc" in history.history else None

    # Construct and return reply Message
    model_record = ArrayRecord(model.get_weights())
    metrics = {"num-examples": len(x_train)}

    if train_loss is not None:
        metrics["train_loss"] = float(train_loss)
    if train_acc is not None:
        metrics["train_acc"] = float(train_acc)
    if train_auc is not None:
        metrics["train_auc"] = float(train_auc)

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local hospital validation data."""

    # Set random seed from config for reproducibility
    random_seed = context.run_config.get("random-seed", 42)
    set_random_seed(random_seed)

    # Load the data for this hospital
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, _, x_val, y_val = load_data_from_bigquery(partition_id, num_partitions)

    # Get feature dimension and load model
    input_dim = get_feature_dim()
    model = load_model(input_dim)

    # Initialize model with received weights
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(ndarrays)

    # Evaluate the model on local validation data
    results = model.evaluate(x_val, y_val, verbose=0)

    # Extract metrics (loss, accuracy, auc, precision, recall)
    loss = results[0]
    accuracy = results[1]
    auc = results[2] if len(results) > 2 else None
    precision = results[3] if len(results) > 3 else None
    recall = results[4] if len(results) > 4 else None

    # Construct and return reply Message
    metrics = {
        "eval_loss": float(loss),
        "eval_acc": float(accuracy),
        "num-examples": len(x_val),
    }

    if auc is not None:
        metrics["eval_auc"] = float(auc)
    if precision is not None:
        metrics["eval_precision"] = float(precision)
    if recall is not None:
        metrics["eval_recall"] = float(recall)

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)
