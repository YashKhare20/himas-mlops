"""
Medical dataset loaders for HIMAS federated learning
Uses sklearn datasets: breast cancer
Loads datasets, splits for hospitals, and prepares for federated learning
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class MedicalDataLoader:
    """Loads and prepares medical datasets for federated learning"""
    
    def __init__(self):
        self.datasets = {
            'breast_cancer': self._load_breast_cancer,
        }
    
    def _load_breast_cancer(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load breast cancer dataset"""
        data = load_breast_cancer()
        return data.data, data.target, data.feature_names.tolist()
    
    def get_dataset(self, dataset_name: str) -> Dict:
        """Get dataset by name"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")
        
        X, y, feature_names = self.datasets[dataset_name]()
        
        logger.info(f"Loaded {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
    
    def split_for_hospitals(self, dataset_name: str, n_hospitals: int = 3) -> List[Dict]:
        """Split dataset across multiple hospitals for federated learning"""
        dataset = self.get_dataset(dataset_name)
        
        X, y = dataset['X'], dataset['y']
        hospital_data = []
        
        # Split data among hospitals
        samples_per_hospital = len(X) // n_hospitals
        
        for i in range(n_hospitals):
            start_idx = i * samples_per_hospital
            if i == n_hospitals - 1:  # Last hospital gets remaining samples
                end_idx = len(X)
            else:
                end_idx = (i + 1) * samples_per_hospital
            
            hospital_X = X[start_idx:end_idx]
            hospital_y = y[start_idx:end_idx]
            
            # Further split into train/test for each hospital
            X_train, X_test, y_train, y_test = train_test_split(
                hospital_X, hospital_y, test_size=0.2, random_state=42
            )
            
            hospital_data.append({
                'hospital_id': f'hospital_{i+1}',
                'X_train': X_train,
                'X_test': X_test, 
                'y_train': y_train,
                'y_test': y_test,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'feature_names': dataset['feature_names']
            })
            
            logger.info(f"Hospital {i+1}: {len(X_train)} train samples, {len(X_test)} test samples")
        
        return hospital_data

# Test the data loader
if __name__ == "__main__":
    loader = MedicalDataLoader()
    
    # Test loading datasets
    cancer_data = loader.get_dataset('breast_cancer')
    print(f"Breast Cancer: {cancer_data['n_samples']} samples, {cancer_data['n_features']} features")
    
    print("Feature names:", cancer_data['feature_names'])

    # Test hospital splitting
    hospital_splits = loader.split_for_hospitals('breast_cancer', n_hospitals=3)
    print(f"Split breast cancer data across {len(hospital_splits)} hospitals")

    #Hospital data: list of dictionaries
    ''' hospital_splits = [
    {hospital_1_data},  # Dictionary for hospital 1
    {hospital_2_data},  # Dictionary for hospital 2
    {hospital_3_data}   # Dictionary for hospital 3 ]'''

    #For each hospital it contains a dictionary with keys : as line 70