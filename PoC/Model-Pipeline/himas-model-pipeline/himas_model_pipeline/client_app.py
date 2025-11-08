"""himas-model-pipeline: A Flower / TensorFlow app for federated ICU mortality prediction."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from himas_model_pipeline.task import load_data_from_bigquery, load_model, get_feature_dim
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import os
import mlflow

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
_MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT_NAME", "himas-federated")
mlflow.set_tracking_uri(_MLFLOW_URI)
mlflow.set_experiment(_MLFLOW_EXP)

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local hospital data."""
    with mlflow.start_run(nested=True, run_name=f"client-train-p{context.node_config['partition-id']}"):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        x_train, y_train, _, _ = load_data_from_bigquery(partition_id, num_partitions)

        input_dim = get_feature_dim()
        model = load_model(input_dim)

        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        model.set_weights(ndarrays)

        epochs = context.run_config["local-epochs"]
        batch_size = context.run_config["batch-size"]
        verbose = context.run_config.get("verbose", 0)

        mlflow.set_tags({
            "role": "client",
            "phase": "train",
            "partition_id": str(partition_id),
            "num_partitions": str(num_partitions),
        })
        mlflow.log_params({
            "local_epochs": epochs,
            "batch_size": batch_size,
        })

        class_weights = compute_class_weight(
            "balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            verbose=verbose,
        )

        train_loss = history.history.get("loss", [None])[-1]
        train_acc = history.history.get("accuracy", [None])[-1]
        train_auc = history.history.get("auc", [None])[-1]

        # SINGLE metrics dict used for BOTH MLflow and Flower
        metrics = {"num-examples": int(len(x_train))}  # ⚠️ key must match server: num-examples
        if train_loss is not None:
            metrics["train_loss"] = float(train_loss)
        if train_acc is not None:
            metrics["train_acc"] = float(train_acc)
        if train_auc is not None:
            metrics["train_auc"] = float(train_auc)

        # Debug safety check (optional but very useful right now)
        print(f"[DEBUG train p{partition_id}] metrics keys: {list(metrics.keys())}")
        assert "num-examples" in metrics, "num-examples missing from train metrics!"

        mlflow.log_metrics(metrics)

        model_record = ArrayRecord(model.get_weights())
        metric_record = MetricRecord(metrics)
        content = RecordDict({"arrays": model_record, "metrics": metric_record})

        return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local hospital validation data."""
    with mlflow.start_run(nested=True, run_name=f"client-eval-p{context.node_config['partition-id']}"):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        _, _, x_val, y_val = load_data_from_bigquery(partition_id, num_partitions)

        input_dim = get_feature_dim()
        model = load_model(input_dim)

        ndarrays = msg.content["arrays"].to_numpy_ndarrays()
        model.set_weights(ndarrays)

        results = model.evaluate(x_val, y_val, verbose=0)

        loss = results[0]
        accuracy = results[1]
        auc = results[2] if len(results) > 2 else None
        precision = results[3] if len(results) > 3 else None
        recall = results[4] if len(results) > 4 else None

        metrics = {
            "eval_loss": float(loss),
            "eval_acc": float(accuracy),
            "num-examples": int(len(x_val)),  # keep same naming convention
        }
        if auc is not None:
            metrics["eval_auc"] = float(auc)
        if precision is not None:
            metrics["eval_precision"] = float(precision)
        if recall is not None:
            metrics["eval_recall"] = float(recall)

        print(f"[DEBUG eval p{partition_id}] metrics keys: {list(metrics.keys())}")
        assert "num-examples" in metrics, "num-examples missing from eval metrics!"

        mlflow.set_tags({
            "role": "client",
            "phase": "evaluate",
            "partition_id": str(partition_id),
        })
        mlflow.log_metrics(metrics)

        metric_record = MetricRecord(metrics)
        content = RecordDict({"metrics": metric_record})

        return Message(content=content, reply_to=msg)
