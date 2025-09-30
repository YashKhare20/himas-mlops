"""
Docker-compatible hospital client for HIMAS
Runs as individual container and connects to central server
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.append('src')

from data.medical_datasets import MedicalDataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import flwr as fl
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HospitalClient(fl.client.NumPyClient):
    """Flower client for Docker deployment"""
    
    def __init__(self, hospital_index: int):
        self.hospital_id = f"hospital_{hospital_index + 1}"
        
        # Load data for this hospital
        loader = MedicalDataLoader()
        all_hospital_data = loader.split_for_hospitals('breast_cancer', n_hospitals=3)
        hospital_data = all_hospital_data[hospital_index]
        
        self.X_train = hospital_data['X_train']
        self.y_train = hospital_data['y_train']
        self.X_test = hospital_data['X_test']
        self.y_test = hospital_data['y_test']
        
        # Create model
        self.model = LogisticRegression(max_iter=5000, random_state=42, warm_start=True)
        self.model.fit(self.X_train[:10], self.y_train[:10])
        
        logger.info(f"Initialized {self.hospital_id}")
        logger.info(f"  Train: {len(self.X_train)}, Test: {len(self.X_test)}")
    
    def get_parameters(self, config):
        return [self.model.coef_.copy(), self.model.intercept_.copy()]
    
    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0].copy()
        self.model.intercept_ = parameters[1].copy()
    
    def fit(self, parameters, config):
        logger.info(f"{self.hospital_id}: Training")
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        
        train_acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
        train_loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        
        logger.info(f"{self.hospital_id}: Accuracy={train_acc:.4f}, Loss={train_loss:.4f}")
        
        return self.get_parameters(config), len(self.X_train), {
            'train_accuracy': float(train_acc),
            'train_loss': float(train_loss)
        }
    
    def evaluate(self, parameters, config):
        logger.info(f"{self.hospital_id}: Evaluating")
        self.set_parameters(parameters)
        
        test_acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
        test_loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        
        logger.info(f"{self.hospital_id}: Test Accuracy={test_acc:.4f}, Loss={test_loss:.4f}")
        
        return float(test_loss), len(self.X_test), {
            'eval_accuracy': float(test_acc),
            'eval_loss': float(test_loss)
        }


def run_hospital_client(hospital_index: int, sleep_time: int = 5):
    """Run hospital client in Docker"""
    logger.info(f"Waiting {sleep_time} seconds for server to start...")
    time.sleep(sleep_time)
    
    logger.info(f"Connecting to server at himas-server:8080")
    
    client = HospitalClient(hospital_index)
    
    fl.client.start_numpy_client(
        server_address='himas-server:8080',
        client=client
    )


if __name__ == "__main__":
    hospital_index = int(sys.argv[1])
    sleep_time = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    run_hospital_client(hospital_index, sleep_time)