import pickle
from pathlib import Path

def load_preprocessing_artifacts(base_path: str):
    """
    Loads scaler.pkl, label_encoders.pkl, feature_order.pkl
    from /app/model/preprocessing/
    """

    artifact_dir = Path(base_path) / "preprocessing"


    with open(artifact_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(artifact_dir / "label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)

    with open(artifact_dir / "feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)

    return scaler, label_encoders, feature_order
