import sys
import os
from pathlib import Path
import torch

# Add the project root and ml backend src to the Python path
# This is necessary to import project modules when running the script directly
project_root = Path(__file__).resolve().parents[4]
ml_backend_src = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(str(ml_backend_src))

# Ensure we can import necessary modules
try:
    from ml.src.api.model_service import ModelService
    from ml.src.models.model_factory import ModelFactory
    from ml.src.models.cnn_lstm_model import CNNLSTMModel
    import logging
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Configure logging to see messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model registry path (should match environment variable or default in model_service.py)
MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "models/registry")

def test_model_loading(symbol: str = "BTC/USDT"):
    logger.info(f"Attempting to load model for symbol: {symbol}")
    
    # Create an instance of ModelService
    # We are mimicking the server's loading logic here
    service = ModelService()
    
    try:
        # Attempt to load the model using the service's method
        # This will internally call torch.load and ModelFactory.create_model
        loaded = service.load_model(symbol)
        
        if loaded:
            logger.info(f"Model for {symbol} loaded successfully!")
            # Optionally, you could add code here to inspect the loaded model or make a dummy prediction
            # For now, just confirming load success is enough.
        else:
            logger.error(f"Failed to load model for {symbol}.")
            
    except Exception as e:
        logger.error(f"An error occurred during model loading: {e}")
        # Print traceback for more detailed error information
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test function
    test_model_loading("BTC/USDT") 