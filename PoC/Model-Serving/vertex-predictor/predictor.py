import os
import pickle
import numpy as np
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
import tensorflow as tf

class CustomPredictor(Predictor):
    def __init__(self):
        return

    def load(self, artifacts_uri: str):
        """Load model and preprocessing artifacts."""
        prediction_utils.download_model_artifacts(artifacts_uri)
        
        # Load model
        self.model = tf.keras.models.load_model("model.keras")
        
        # Load preprocessing artifacts - FIX: double preprocessing folder
        with open("preprocessing/preprocessing/feature_order.pkl", "rb") as f:
            self.feature_order = pickle.load(f)
        with open("preprocessing/preprocessing/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open("preprocessing/preprocessing/label_encoders.pkl", "rb") as f:
            self.label_encoders = pickle.load(f)

    def preprocess(self, prediction_input: dict):
        """Preprocess raw input into model-ready format."""
        instances = prediction_input.get("instances", [])
        
        processed_batch = []
        for instance in instances:
            processed = []
            for feature in self.feature_order:
                if feature not in instance:
                    processed.append(0)
                    continue
                
                value = instance[feature]
                
                # Numeric features
                if feature in self.scaler.feature_names_in_:
                    processed.append(float(value))
                # Categorical features
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
            categorical = processed[num_feature_count:]
            
            processed_batch.append(scaled_numeric + categorical)
        
        return np.array(processed_batch)

    def predict(self, instances):
        """Run prediction on preprocessed data."""
        return self.model.predict(instances)

    def postprocess(self, prediction_results):
        """Convert model output to response format."""
        predictions = []
        for prob in prediction_results:
            predictions.append({
                "probability": float(prob[0]),
                "prediction": int(prob[0] >= 0.5)
            })
        return {"predictions": predictions}
