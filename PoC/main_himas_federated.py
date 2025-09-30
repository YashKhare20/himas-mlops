"""
HIMAS Layer 2: Complete Federated Healthcare Intelligence System
Integrates all components: LangChain agents + ChromaDB + MLflow + Flower federated learning
"""

import logging
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.medical_datasets import MedicalDataLoader
from src.federated.flower_server import HIMASFederatedCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HIMAS')

class CompleteFederatedHIMAS:
    """Complete HIMAS Layer 2 system with federated learning"""
    
    def __init__(self):
        self.data_loader = MedicalDataLoader()
        self.coordinator = HIMASFederatedCoordinator(num_rounds=3)
        
        logger.info("HIMAS Federated Healthcare Intelligence System initialized")
    
    def demonstrate_complete_system(self):
        """Demonstrate the complete federated HIMAS system"""
        
        print("=" * 60)
        print("HIMAS Layer 2: Federated Healthcare Intelligence System")
        print("=" * 60)
        
        # Step 1: Load medical datasets
        print("\n1. Loading Medical Datasets...")
        bc_data = self.data_loader.get_dataset('breast_cancer')
        
        print(f"   - Breast Cancer: {bc_data['n_samples']} samples, {bc_data['n_features']} features")
        
        # Step 2: Split data across hospitals
        print("\n2. Simulating 3 Hospital Networks...")
        hospital_data = self.data_loader.split_for_hospitals('breast_cancer', n_hospitals=3)
        
        for hospital in hospital_data:
            print(f"   - {hospital['hospital_id']}: {hospital['n_train_samples']} train, {hospital['n_test_samples']} test")
        
        
        # Step 3: Run federated learning
        print("\n3. Running Federated Learning Simulation...")
        print("   Hospitals collaborating on breast cancer diagnosis...")

        history = self.coordinator.run_simulation(hospital_data)

        accuracies = history.metrics_distributed["federated_accuracy"]
        print(f"   Round 1 Accuracy: {accuracies[0][1]:.2f}")
        print(f"   Final Accuracy: {accuracies[-1][1]:.2f}")
        
        improvement = accuracies[-1][1] - accuracies[0][1]
        print(f"   Improvement: {improvement:+.2f}")
        
        print("   Federated learning completed successfully")
        
        print("\n" + "=" * 60)
        print("HIMAS Layer 2 PoC Successfully Demonstrated!")
        print("=" * 60)

def main():
    """Main entry point for HIMAS federated system demonstration"""
    
    try:
        # Create and run complete system
        himas = CompleteFederatedHIMAS()
        himas.demonstrate_complete_system()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"Error running HIMAS system: {e}")

if __name__ == "__main__":
    main()