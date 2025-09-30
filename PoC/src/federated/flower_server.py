"""
Flower federated learning server for HIMAS
Coordinates federated learning across multiple hospitals
Uses Flower 1.11.0 API
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Flower imports
import flwr as fl
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, Metrics, Parameters, ndarrays_to_parameters

logger = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from all hospitals"""
    accuracies = [num_examples * m.get("eval_accuracy", 0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, m in metrics]
    
    if not examples or sum(examples) == 0:
        return {}
    
    aggregated_accuracy = sum(accuracies) / sum(examples)
    
    logger.info(f"Federated evaluation - Aggregated accuracy: {aggregated_accuracy:.4f}")
    logger.info(f"   Total examples: {sum(examples)}")
    
    return {"federated_accuracy": aggregated_accuracy}


def get_initial_parameters() -> Parameters:
    """Initialize global model parameters"""
    from sklearn.linear_model import LogisticRegression
    
    n_features = 30
    model = LogisticRegression(max_iter=5000, random_state=42)
    
    dummy_X = np.random.randn(10, n_features)
    dummy_y = np.random.randint(0, 2, 10)
    model.fit(dummy_X, dummy_y)
    
    initial_params = [model.coef_, model.intercept_]
    
    logger.info(f"Initialized global model")
    logger.info(f"  Coef shape: {initial_params[0].shape}")
    logger.info(f"  Intercept shape: {initial_params[1].shape}")
    
    return ndarrays_to_parameters(initial_params)


def get_federated_strategy() -> FedAvg:
    """Create federated learning strategy"""
    initial_parameters = get_initial_parameters()
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    logger.info("Created FedAvg strategy")
    return strategy


# Create Flower ServerApp
app = ServerApp()


@app.main()
def main(context: Context) -> None:
    """Main server application entry point"""
    logger.info("\n" + "="*70)
    logger.info("HIMAS Federated Learning Server")
    logger.info("="*70)
    
    num_rounds = context.run_config.get("num-server-rounds", 3)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"   - Server rounds: {num_rounds}")
    logger.info(f"   - Strategy: FedAvg")
    logger.info(f"   - Min hospitals: 2")
    
    strategy = get_federated_strategy()
    
    logger.info(f"\nStarting federated learning...")
    logger.info("="*70 + "\n")
    
    config = ServerConfig(num_rounds=num_rounds)


class HIMASFederatedCoordinator:
    """Coordinates federated learning simulation"""
    
    def __init__(self, num_rounds: int = 3):
        self.num_rounds = num_rounds
    
    def run_simulation(self, hospital_data_list: List[Dict]):
        """Run federated learning simulation"""
        logger.info("\n" + "="*70)
        logger.info("HIMAS Federated Learning Simulation")
        logger.info("="*70)
        logger.info(f"   Hospitals: {len(hospital_data_list)}")
        logger.info(f"   Rounds: {self.num_rounds}")
        
        from .flower_client import HIMASFlowerClient, hospital_data_cache
        
        # FIXED: Use Context parameter
        def client_fn(context: Context):
            """Create client for hospital"""
            cid = context.node_config.get("partition-id", 0)
            hospital_idx = int(cid)
            return HIMASFlowerClient(partition_id=hospital_idx).to_client()
        
        # Populate cache
        for idx, hospital_data in enumerate(hospital_data_list):
            hospital_data_cache[idx] = hospital_data
        
        logger.info(f"\nStarting simulation...")
        logger.info("="*70 + "\n")
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(hospital_data_list),
            config=ServerConfig(num_rounds=self.num_rounds),
            strategy=get_federated_strategy(),
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )
        
        logger.info("\n" + "="*70)
        logger.info("Simulation completed!")
        logger.info("="*70 + "\n")
        
        return history


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    from data.medical_datasets import MedicalDataLoader
    
    print("\n" + "="*70)
    print("HIMAS Federated Learning Server - Test Mode")
    print("="*70)
    
    loader = MedicalDataLoader()
    hospital_data_list = loader.split_for_hospitals('breast_cancer', n_hospitals=3)
    
    print(f"\nPrepared data for {len(hospital_data_list)} hospitals:")
    for hospital_data in hospital_data_list:
        print(f"   - {hospital_data['hospital_id']}: "
              f"{hospital_data['n_train_samples']} train, "
              f"{hospital_data['n_test_samples']} test")
    
    print(f"\nRunning simulation...")
    print("="*70 + "\n")
    
    coordinator = HIMASFederatedCoordinator(num_rounds=3)
    
    try:
        history = coordinator.run_simulation(hospital_data_list)
        
        print("\n" + "="*70)
        print("Simulation completed successfully!")
        print("="*70)
        print("\nFederated model training complete!")
        print("All 3 hospitals collaborated without sharing patient data.")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()