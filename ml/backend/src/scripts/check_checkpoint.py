import torch
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint file not found at {model_path}")
        sys.exit(1)

    try:
        checkpoint = torch.load(model_path)
        model_type = checkpoint.get('model_type', 'Not Found')
        print(f"Model type in checkpoint: {model_type}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1) 