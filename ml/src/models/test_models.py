"""
Test script for model architectures

This script validates that all model architectures function correctly,
including proper device handling for Apple Silicon (MPS).
"""

import os
import sys
import torch
import time
import logging
from pathlib import Path

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import (
    BaseModel, 
    DirectionalLoss, 
    ModelFactory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dummy_data(batch_size=8, seq_len=50, features=10):
    """
    Generate dummy data for testing models
    
    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        features: Number of features
        
    Returns:
        X: Input tensor
        y: Target tensor
    """
    # Generate random input data
    X = torch.randn(batch_size, seq_len, features)
    
    # Generate random target data
    y = torch.randn(batch_size, 5, 1)  # 5-step forecast, 1 output feature
    
    return X, y

def test_model(model_type, input_dim=10, output_dim=1, seq_len=50, batch_size=8, forecast_horizon=5):
    """
    Test a specific model architecture
    
    Args:
        model_type: Type of model to test
        input_dim: Number of input features
        output_dim: Number of output features
        seq_len: Length of input sequences
        batch_size: Batch size for testing
        forecast_horizon: Number of steps to forecast
    """
    print(f"\n{'='*80}")
    print(f"Testing {model_type} model...")
    print(f"{'='*80}")
    
    # Generate dummy data
    X, y = generate_dummy_data(batch_size, seq_len, input_dim)
    
    # Record time to measure initialization speed
    start_time = time.time()
    
    # Create model using factory
    model = ModelFactory.create_model(
        model_type=model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        hidden_dim=64,
        num_layers=2
    )
    
    init_time = time.time() - start_time
    print(f"Model initialization time: {init_time:.4f} seconds")
    
    # Check device
    print(f"Model device: {model.device}")
    
    # Record time for forward pass
    start_time = time.time()
    
    # Test forward pass
    with torch.no_grad():
        output = model(X.to(model.device))
    
    forward_time = time.time() - start_time
    print(f"Forward pass time: {forward_time:.4f} seconds")
    
    # Check output shape
    expected_shape = (batch_size, forecast_horizon, output_dim)
    if output.shape == expected_shape:
        print(f"✅ Output shape is correct: {output.shape}")
    else:
        print(f"❌ Output shape is wrong: {output.shape}, expected: {expected_shape}")
    
    # Test directional loss
    loss_fn = DirectionalLoss()
    pred = output.cpu()
    target = y
    
    start_time = time.time()
    loss = loss_fn(pred, target)
    loss_time = time.time() - start_time
    
    print(f"Directional loss value: {loss.item():.4f}, calculation time: {loss_time:.4f} seconds")
    
    # Create a temporary directory for saving models
    temp_dir = "temp_models"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{model_type}_model.pt")
    
    # Test save and load
    model.save(temp_path)
    print(f"Model saved to {temp_path}")
    
    # Test loading
    try:
        loaded_model = BaseModel.load(temp_path)
        print(f"Model loaded from {temp_path}")
        print(f"Loaded model type: {type(loaded_model).__name__}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    
    # Return model and timing data
    return {
        "model_type": model_type,
        "init_time": init_time,
        "forward_time": forward_time,
        "loss_time": loss_time,
        "total_time": init_time + forward_time + loss_time
    }

def main():
    """Run tests for all model architectures"""
    print("\nTesting PyTorch model architectures with MPS support...\n")
    
    # Check available devices
    device_info = (
        f"MPS (Apple Silicon): {torch.backends.mps.is_available()}\n"
        f"CUDA (NVIDIA): {torch.cuda.is_available()}\n"
    )
    print(f"Available devices:\n{device_info}")
    
    # Test parameters
    input_dim = 20  # Number of features (technical indicators, etc.)
    output_dim = 1  # Price prediction
    seq_len = 60    # 60 timesteps of data (e.g., 1 hour candles for 60 hours)
    forecast_horizon = 5  # Predict 5 steps ahead
    batch_size = 16   # Batch size for testing
    
    # Model types to test
    model_types = ['lstm', 'gru', 'transformer', 'cnn_lstm']
    
    # Run tests
    results = []
    for model_type in model_types:
        try:
            result = test_model(
                model_type=model_type,
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                batch_size=batch_size,
                forecast_horizon=forecast_horizon
            )
            results.append(result)
        except Exception as e:
            print(f"Error testing {model_type} model: {str(e)}")
            results.append({
                "model_type": model_type,
                "init_time": 0,
                "forward_time": 0,
                "loss_time": 0,
                "total_time": 0,
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("Performance Summary:")
    print("="*80)
    print(f"{'Model Type':<15} {'Init Time (s)':<15} {'Forward Time (s)':<20} {'Total Time (s)':<15}")
    print("-"*80)
    
    for result in results:
        if "error" in result:
            print(f"{result['model_type']:<15} ERROR: {result['error']}")
        else:
            print(f"{result['model_type']:<15} {result['init_time']:<15.4f} {result['forward_time']:<20.4f} {result['total_time']:<15.4f}")
    
    # Clean up
    import shutil
    if os.path.exists("temp_models"):
        shutil.rmtree("temp_models")
        print("\nRemoved temporary models directory")

if __name__ == "__main__":
    main() 