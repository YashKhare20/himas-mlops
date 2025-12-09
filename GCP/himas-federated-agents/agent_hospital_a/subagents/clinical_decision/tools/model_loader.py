"""
Model & Preprocessor Loader (Fits Preprocessor on First Load)
"""

import logging
import tensorflow as tf
from google.cloud import storage
import tempfile
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global caches
_MODEL_CACHE = {}
_PREPROCESSOR_CACHE = {}

# Configure TensorFlow for single-threaded operation (prevents deadlocks)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Force eager execution
tf.config.run_functions_eagerly(True)


class ModelLoader:
    """
    Loads model and creates fitted preprocessor.

    **Preprocessor Strategy:**
    Since saved preprocessor not available, we fit it on BigQuery training data
    when the agent first starts. This ensures EXACT same preprocessing as training.
    """

    def __init__(self, model_gcs_path: str, project_id: str, dataset_id: str):
        """
        Initialize loader.

        Args:
            model_gcs_path: GCS path to .keras model
            project_id: GCP project ID (for BigQuery)
            dataset_id: BigQuery dataset (for fitting preprocessor)
        """
        self.model_gcs_path = model_gcs_path
        self.project_id = project_id
        self.dataset_id = dataset_id

        # Parse GCS path
        if model_gcs_path.startswith('gs://'):
            parts = model_gcs_path.replace('gs://', '').split('/', 1)
            self.model_bucket = parts[0]
            self.model_blob = parts[1] if len(parts) > 1 else None
        else:
            raise ValueError(f"Invalid GCS path: {model_gcs_path}")

    def load_model(self) -> tf.keras.Model:
        """Loads model from GCS with caching."""
        if self.model_gcs_path in _MODEL_CACHE:
            logger.info("Model loaded from cache")
            return _MODEL_CACHE[self.model_gcs_path]

        try:
            logger.info(f"Downloading model from {self.model_gcs_path}")

            storage_client = storage.Client()
            bucket = storage_client.bucket(self.model_bucket)
            blob = bucket.blob(self.model_blob)

            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, 'model.keras')
                blob.download_to_filename(local_path)

                model = tf.keras.models.load_model(local_path)
                _MODEL_CACHE[self.model_gcs_path] = model

                logger.info(f"Model loaded: input_shape={model.input_shape}")
                return model

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def load_preprocessor(self):
        """
        Creates and fits preprocessor on BigQuery training data.

        **IMPORTANT:** This replicates the exact preprocessing from training
        by fitting on the same training data the model was trained on.

        Returns:
            Fitted DataPreprocessor instance
        """
        cache_key = f"{self.project_id}_{self.dataset_id}_preprocessor"

        if cache_key in _PREPROCESSOR_CACHE:
            logger.info("Preprocessor loaded from cache")
            return _PREPROCESSOR_CACHE[cache_key]

        try:
            # Import preprocessor
            from ....utils.data_preprocessor import fit_preprocessor_on_bigquery_training_data

            logger.info("Fitting preprocessor on BigQuery training data...")
            logger.info("This happens ONCE on agent startup, then cached")

            # Fit on combined training data (same as evaluation does)
            preprocessor = fit_preprocessor_on_bigquery_training_data(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                hospital_id='all'  # Fit on all hospitals like evaluation does
            )

            _PREPROCESSOR_CACHE[cache_key] = preprocessor

            logger.info("Preprocessor fitted and cached")
            logger.info(f"Feature dim: {preprocessor.feature_dim}")

            return preprocessor

        except Exception as e:
            logger.error(f"Preprocessor loading failed: {str(e)}")
            raise RuntimeError(f"Preprocessor loading failed: {str(e)}")

    def load_all(self):
        """
        Loads both model and preprocessor.

        Returns:
            Tuple of (model, preprocessor)
        """
        model = self.load_model()
        preprocessor = self.load_preprocessor()
        return model, preprocessor


# Singleton instance
_loader_instance = None


def get_model_loader(
    model_gcs_path: str = None,
    project_id: str = None,
    dataset_id: str = None
) -> ModelLoader:
    """Gets singleton ModelLoader instance."""
    global _loader_instance

    if _loader_instance is None:
        if model_gcs_path is None:
            from ....config import config
            model_gcs_path = config.MODEL_GCS_PATH
            project_id = config.GOOGLE_CLOUD_PROJECT
            dataset_id = config.BIGQUERY_DATASET

        _loader_instance = ModelLoader(model_gcs_path, project_id, dataset_id)

    return _loader_instance
