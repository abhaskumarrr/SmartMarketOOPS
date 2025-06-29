import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import os
import pandas as pd
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the model and feature builder
from ml.src.training.train_hybrid_model import HybridModel
from ml.src.features.build_features import build_features

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/models/hybrid_model.pth')
INPUT_DIM = 10 # Based on the features in build_features.py

# --- Load Model ---
model = HybridModel(INPUT_DIM)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Hybrid model loaded successfully.")
else:
    print(f"Warning: Hybrid model not found at {MODEL_PATH}")


# --- FastAPI App ---
app = FastAPI(title="Advanced Hybrid Decision Engine")

class MarketData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

@app.post("/get_decision")
async def get_decision(data: list[MarketData]):
    """
    Makes a trading decision based on the hybrid model and a sequence of market data.
    """
    # 1. Create a DataFrame from the input data
    df = pd.DataFrame([d.dict() for d in data])

    # 2. Build features
    features_df = build_features(df)
    
    if features_df.empty:
        return {"decision": "hold", "reason": "Not enough data to build features"}

    # Use the most recent feature set for the decision
    latest_features = features_df.iloc[-1]
    
    # 3. Expert Rules (Simplified)
    if latest_features['volume_change'] > 5.0: # Example rule: avoid extreme volume spikes
        return {"decision": "hold", "reason": "Extreme volume spike detected"}

    # 4. Prepare features for the model
    model_features = latest_features.drop(['regime', 'bias']).values
    features_tensor = torch.FloatTensor(np.array(model_features).reshape(1, -1))

    # 5. Get prediction from the model
    with torch.no_grad():
        output = model(features_tensor)
        decision_int = torch.argmax(output, dim=1).item()

    decision_map = {0: "hold", 1: "buy", 2: "sell"}
    final_decision = decision_map.get(decision_int, "hold")
    
    return {
        "decision": final_decision,
        "reason": "Decision from hybrid model",
        "model_output": decision_int,
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Analysis-Execution Bridge"}
