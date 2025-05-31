#!/usr/bin/env python3
"""
Simple model test to verify enhanced system is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing enhanced model creation...")
    
    from src.models.model_factory import ModelFactory
    print("✅ ModelFactory imported successfully")
    
    # Test enhanced transformer creation
    model = ModelFactory.create_model(
        model_type='enhanced_transformer',
        input_dim=20,
        output_dim=1,
        seq_len=100,
        forecast_horizon=1,
        d_model=64,  # Smaller for testing
        nhead=4,
        num_layers=2
    )
    print("✅ Enhanced Transformer model created successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test model registry
    from src.models.model_registry import ModelRegistry
    registry = ModelRegistry()
    print("✅ Model Registry initialized successfully")
    
    print("\n🎉 Enhanced system components are working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
