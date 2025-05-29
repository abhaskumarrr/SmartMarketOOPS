"""
Environment configuration for the ML service.

This module loads and validates environment variables required by the ML service.
It also provides default values for development environments.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import re
from dotenv import load_dotenv

# Find the project root and load the .env file
project_root = Path(__file__).parent.parent.parent.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    logging.info(f"Loaded environment from {env_path}")
else:
    logging.warning(f"No .env file found at {env_path}. Using default values.")

# Environment configuration with defaults
env_config = {
    # Service configuration
    "NODE_ENV": os.getenv("NODE_ENV", "development"),
    "ML_PORT": int(os.getenv("ML_PORT", "3002")),
    "ML_HOST": os.getenv("ML_HOST", "0.0.0.0"),
    
    # Database connection
    "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smoops?schema=public"),
    
    # ML model configuration
    "MODEL_DIR": os.getenv("MODEL_DIR", str(project_root / "ml" / "models")),
    "DATA_DIR": os.getenv("DATA_DIR", str(project_root / "ml" / "data")),
    "LOG_DIR": os.getenv("LOG_DIR", str(project_root / "ml" / "logs")),
    
    # PyTorch configuration
    "TORCH_MPS_ENABLE": os.getenv("TORCH_MPS_ENABLE", "1") == "1",  # Apple Silicon GPU acceleration
    "TORCH_DEVICE": os.getenv("TORCH_DEVICE", "mps"),  # mps (Apple), cuda (NVIDIA), or cpu
    
    # Logging configuration
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    "PYTHONUNBUFFERED": os.getenv("PYTHONUNBUFFERED", "1") == "1",
    
    # API keys (for external services)
    "DELTA_EXCHANGE_API_KEY": os.getenv("DELTA_EXCHANGE_API_KEY", ""),
    "DELTA_EXCHANGE_API_SECRET": os.getenv("DELTA_EXCHANGE_API_SECRET", ""),
    "DELTA_EXCHANGE_TESTNET": os.getenv("DELTA_EXCHANGE_TESTNET", "true").lower() == "true",
}

def validate_env() -> Dict[str, str]:
    """
    Validate the environment configuration and return any errors.
    
    Returns:
        Dict[str, str]: Dictionary of environment variable names to error messages
    """
    errors = {}
    
    # Check required variables for production
    if env_config["NODE_ENV"] == "production":
        # In production, API keys are required
        if not env_config["DELTA_EXCHANGE_API_KEY"]:
            errors["DELTA_EXCHANGE_API_KEY"] = "Missing API key in production environment"
        if not env_config["DELTA_EXCHANGE_API_SECRET"]:
            errors["DELTA_EXCHANGE_API_SECRET"] = "Missing API secret in production environment"
    
    # Validate DATABASE_URL format
    db_url = env_config["DATABASE_URL"]
    db_url_pattern = r'^postgresql://[^:]+:[^@]+@[^:]+:\d+/[^?]+(\\?.*)?$'
    if not re.match(db_url_pattern, db_url):
        errors["DATABASE_URL"] = "Invalid database URL format"
    
    # Validate port number
    port = env_config["ML_PORT"]
    if not isinstance(port, int) or port < 1 or port > 65535:
        errors["ML_PORT"] = f"Invalid port number: {port}. Must be between 1-65535."
    
    # Validate pytorch device
    device = env_config["TORCH_DEVICE"]
    valid_devices = ["mps", "cuda", "cpu"]
    if device not in valid_devices:
        errors["TORCH_DEVICE"] = f"Invalid device: {device}. Must be one of {valid_devices}"
    
    return errors

def get_env() -> Dict[str, Any]:
    """
    Get the environment configuration.
    
    Returns:
        Dict[str, Any]: The environment configuration
    """
    return env_config

def setup_logging():
    """Configure logging based on environment variables"""
    log_level_name = env_config["LOG_LEVEL"].upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set up file logging if in production
    if env_config["NODE_ENV"] == "production":
        log_dir = Path(env_config["LOG_DIR"])
        log_dir.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_dir / "ml_service.log")
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)

# Run validation on module import
errors = validate_env()

# Print validation errors
if errors:
    for var_name, error_msg in errors.items():
        logging.error(f"Environment error - {var_name}: {error_msg}")
    
    # Exit in production mode if there are errors
    if env_config["NODE_ENV"] == "production":
        logging.critical("Exiting due to environment configuration errors in production mode")
        sys.exit(1)

# Set up logging
setup_logging()

# Log environment in development mode (with sensitive info masked)
if env_config["NODE_ENV"] == "development":
    safe_config = env_config.copy()
    if safe_config["DELTA_EXCHANGE_API_KEY"]:
        safe_config["DELTA_EXCHANGE_API_KEY"] = "********"
    if safe_config["DELTA_EXCHANGE_API_SECRET"]:
        safe_config["DELTA_EXCHANGE_API_SECRET"] = "********"
    
    # Extract username/password from DATABASE_URL for masking
    db_url = safe_config["DATABASE_URL"]
    masked_db_url = re.sub(r'postgresql://([^:]+):([^@]+)@', 'postgresql://****:****@', db_url)
    safe_config["DATABASE_URL"] = masked_db_url
    
    logging.info(f"ML Service Environment: {safe_config}")

# Export the config
env = get_env() 