import os
import pickle
import numpy as np
import keras

from preprocessing_loader import load_preprocessing_artifacts


class Predictor:
    def __init__(self):
        """
        Loads:
        - model
        - scaler
        - label encoders
        - feature order
        """
        artifacts_path = "/model"

        self.model = keras.models.load_model(f"{artifacts_path}/model.keras")
        (self.scaler,
         self.label_encoders,
         self.feature_order) = load_preprocessing_artifacts(artifacts_path)

    def preprocess(self, input_dict: dict):
        """Convert raw input into model-ready numerical array."""

        processed = []

        for feature in self.feature_order:

            if feature not in input_dict:
                processed.append(0)  # simple fallback
                continue

            value = input_dict[feature]

            # numeric
            if feature in self.scaler.feature_names_in_:
                processed.append(float(value))

            # categorical
            elif feature in self.label_encoders:
                le = self.label_encoders[feature]
                processed.append(
                    le.transform([value])[0] if value in le.classes_ else -1
                )

            else:
                processed.append(0)

        # Scale numeric features
        num_feature_count = len(self.scaler.feature_names_in_)
        numeric = np.array(processed[:num_feature_count]).reshape(1, -1)
        scaled_numeric = self.scaler.transform(numeric)[0].tolist()

        # Keep categorical as is
        categorical = processed[num_feature_count:]

        return np.array([scaled_numeric + categorical])

    def predict(self, payload: dict):
        X = self.preprocess(payload)
        prob = self.model.predict(X)[0][0]
        return 1 if prob >= 0.5 else 0
