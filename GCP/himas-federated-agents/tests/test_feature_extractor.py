import logging
import sys
import os

# Configure logging to show debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.getcwd())

from agent_hospital_a.utils.feature_extractor import FeatureExtractor

def test_feature_extractor():
    logger.info("Testing FeatureExtractor...")
    
    # Test cases
    test_cases = [
        ("SATURDAY_NIGHT", 1, 1),  # Weekend=1, Night=1
        ("Monday Morning", 0, 0),  # Weekend=0, Night=0
        ("2025-11-29T17:37:28", 1, 0), # Saturday (Weekend=1), 5PM (Night=0)
        ("2025-11-30T02:00:00", 1, 1), # Sunday (Weekend=1), 2AM (Night=1)
        (None, -1, -1) # Should default to current time (dynamic)
    ]
    
    for input_str, expected_weekend, expected_night in test_cases:
        logger.info(f"Testing input: {input_str}")
        
        # Test is_weekend
        is_weekend = FeatureExtractor.is_weekend(input_str)
        logger.info(f"  is_weekend: {is_weekend}")
        if expected_weekend != -1:
            assert is_weekend == expected_weekend, f"Expected weekend={expected_weekend} for {input_str}, got {is_weekend}"
            
        # Test is_night_admission
        is_night = FeatureExtractor.is_night_admission(input_str)
        logger.info(f"  is_night: {is_night}")
        if expected_night != -1:
            assert is_night == expected_night, f"Expected night={expected_night} for {input_str}, got {is_night}"

if __name__ == "__main__":
    try:
        test_feature_extractor()
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
