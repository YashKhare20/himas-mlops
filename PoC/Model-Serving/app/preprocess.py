"""
Inference Preprocessing Component

This is the serving-time version of the preprocessing logic.
Training/evaluation uses DataPreprocessor (fit() + transform()).
Serving uses InferencePreprocessor (load() + transform()).
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path


class InferencePreprocessor:
    def __init__(self, artifact_dir: str):
        """
        Load pre-fitted artifacts needed for inference:
        - scaler.pkl
        - label_encoders.pkl
        - feature_order.pkl
        """
        artifact_dir = Path(artifact_dir)

        self.scaler = self._load_pickle(artifact_dir / "scaler.pkl")
        self.encoders = self._load_pickle(artifact_dir / "label_encoders.pkl")
        self.feature_order = self._load_pickle(artifact_dir / "feature_order.pkl")

    def _load_pickle(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessing artifact: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Takes a pandas DataFrame of raw inputs (one or more rows)
        and converts it into the exact model-ready format.
        """

        df = df.copy()

        # 1. Ensure all expected columns exist
        for col in self.feature_order:
            if col not in df:
                df[col] = None

        # Separate numeric and categorical columns using scaler info
        num_cols = list(self.scaler.feature_names_in_)
        num_df = df[num_cols].astype(float).fillna(df[num_cols].median())

        # 2. Scale numeric features
        X_num = self.scaler.transform(num_df)

        # 3. Encode categorical features
        cat_arrays = []
        for col, encoder in self.encoders.items():
            values = df[col].astype(str).fillna("Unknown")
            encoded = np.array([
                encoder.transform([v])[0] if v in encoder.classes_ else -1
                for v in values
            ])
            cat_arrays.append(encoded.reshape(-1, 1))

        # 4. Combine numeric + categorical
        if cat_arrays:
            X_cat = np.hstack(cat_arrays)
            final = np.hstack([X_num, X_cat])
        else:
            final = X_num

        return final
