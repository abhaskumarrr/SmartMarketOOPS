"""
Configuration settings for the ML pipeline
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the root .env file
dotenv_path = Path(__file__).parents[3] / '.env'
load_dotenv(dotenv_path)

# API configuration
API_CONFIG = {
    "delta_exchange": {
        "base_url": "https://testnet-api.delta.exchange" if os.getenv("DELTA_EXCHANGE_TESTNET", "true").lower() == "true" else "https://api.delta.exchange",
        "api_key": os.getenv("DELTA_EXCHANGE_API_KEY", "HmerKHhySssgFIAfEIh4CYA5E3VmKg"),
        "api_secret": os.getenv("DELTA_EXCHANGE_API_SECRET", "1YNVg1x9cIjz1g3BPOQPUJQr6LhEm8w7cTaXi8ebJYPUpx5BKCQysMoLd6FT"),
        "testnet": os.getenv("DELTA_EXCHANGE_TESTNET", "true").lower() == "true"
    }
}

# Data settings
DATA_CONFIG = {
    "symbols": ["BTCUSD", "ETHUSD"],  # Default symbols to track
    "timeframes": ["1h", "4h", "1d"],  # Default timeframes for data collection
    "features": [
        "open", "high", "low", "close", "volume",
        "funding_rate", "open_interest", "mark_price"
    ],
    "train_test_split": 0.8,
    "validation_split": 0.2,
    "sequence_length": 48,  # 48 hours of historical data for prediction
    "target_columns": ["close"],  # Default target for prediction
    "normalize": True,
    "data_dir": str(Path(__file__).parents[2] / "data"),
    "raw_data_dir": str(Path(__file__).parents[2] / "data" / "raw"),
    "processed_data_dir": str(Path(__file__).parents[2] / "data" / "processed")
}

# Model settings
MODEL_CONFIG = {
    "model_type": "lstm",  # Default model type
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10,  # Early stopping patience
    "device": "mps",  # Use Metal Performance Shaders for Apple Silicon
    "model_dir": str(Path(__file__).parents[2] / "models"),
    "tensorboard_dir": str(Path(__file__).parents[2] / "logs" / "tensorboard")
}

# Training settings
TRAINING_CONFIG = {
    "optimizer": "adam",
    "loss_function": "mse",
    "metrics": ["mae", "rmse", "r2"],
    "scheduler": "reduce_lr_on_plateau",
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "save_best_only": True,
    "cross_validation": False,
    "n_splits": 5  # Number of CV folds if cross_validation is True
}

# Combined config
CONFIG = {
    "api": API_CONFIG,
    "data": DATA_CONFIG,
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG
}

def save_config(config=CONFIG, filepath=None):
    """Save the configuration to a JSON file"""
    if filepath is None:
        filepath = str(Path(__file__).parents[2] / "config.json")
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    
    return filepath

def load_config(filepath=None):
    """Load configuration from a JSON file"""
    if filepath is None:
        filepath = str(Path(__file__).parents[2] / "config.json")
    
    if not os.path.exists(filepath):
        # If no config file exists, save the default one
        save_config(CONFIG, filepath)
        return CONFIG
    
    with open(filepath, 'r') as f:
        return json.load(f)

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for the ML pipeline"""
    directories = [
        DATA_CONFIG["data_dir"],
        DATA_CONFIG["raw_data_dir"],
        DATA_CONFIG["processed_data_dir"],
        MODEL_CONFIG["model_dir"],
        MODEL_CONFIG["tensorboard_dir"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize directories
create_directories()

if __name__ == "__main__":
    # If run directly, save the default config
    config_path = save_config()
    print(f"Default configuration saved to {config_path}") 