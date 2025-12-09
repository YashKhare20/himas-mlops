import os
import pickle
import numpy as np
import keras
from preprocessing_loader import load_preprocessing_artifacts

class Predictor:
    def __init__(self):
        artifacts_path = "/app/local_model"
        model_path = f"{artifacts_path}/model.keras"
        print(f"Loading model from: {model_path}")
        
        self.model = keras.models.load_model(model_path)
        (self.scaler,
         self.label_encoders,
         self.feature_order) = load_preprocessing_artifacts(artifacts_path)
        
        print("✅ Model loaded")
        print(f"Expected features: {len(self.feature_order)}")
        print(f"Feature order: {self.feature_order[:5]}...")  # Show first 5

    def preprocess(self, input_dict: dict):
        """Convert raw input into model-ready numerical array."""
        # Separate numeric and categorical values
        numeric_values = []
        categorical_values = []
        
        for feature in self.feature_order:
            value = input_dict.get(feature, None)
            
            # Numeric features
            if feature in self.scaler.feature_names_in_:
                if value is None:
                    numeric_values.append(0)  # Use 0 for missing (should be median ideally)
                else:
                    numeric_values.append(float(value))
            
            # Categorical features
            elif feature in self.label_encoders:
                le = self.label_encoders[feature]
                if value is None:
                    categorical_values.append(-1)  # Missing category
                elif value in le.classes_:
                    categorical_values.append(le.transform([value])[0])
                else:
                    categorical_values.append(-1)  # Unknown category
            else:
                # Feature not in scaler or encoders
                categorical_values.append(0)
        
        # Scale numeric features
        numeric_array = np.array(numeric_values).reshape(1, -1)
        scaled_numeric = self.scaler.transform(numeric_array)[0].tolist()
        
        # Combine scaled numeric + encoded categorical
        result = np.array([scaled_numeric + categorical_values])
        
        return result
    def predict(self, payload: dict):
        X = self.preprocess(payload)
        prob = self.model.predict(X)[0][0]
        print(f"Raw probability: {prob}")  # DEBUG
        return 1 if prob >= 0.5 else 0
