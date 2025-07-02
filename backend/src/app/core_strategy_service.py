from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from . import models, schemas, services
from .database import get_db
from .auth import get_current_user
from .core_trading_strategy import CoreTradingStrategy

# --- Pydantic Models for API Data Validation ---

class MarketData(BaseModel):
    daily: dict
    h4: dict
    m15: dict

class SignalResponse(BaseModel):
    signal: Literal['buy', 'sell', 'hold']
    confidence: float
    reason: str
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None

# --- API Router ---

router = APIRouter(
    prefix="/strategy",
    tags=["Core Trading Strategy"],
)

# Instantiate our strategy brain once to be used by the API
strategy = CoreTradingStrategy()

def convert_numpy_types(data):
    """Recursively converts numpy types in a dictionary to standard Python types."""
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(i) for i in data]
    elif isinstance(data, (np.bool_, np.integer, np.floating)):
        return data.item()
    return data

@router.post("/predict", response_model=SignalResponse)
async def get_prediction(
    data: MarketData,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Takes daily, 4-hour, and 15-minute market data, processes it through the 
    CoreTradingStrategy, saves the signal, and returns the result.
    """
    try:
        # Convert the incoming dictionaries to pandas DataFrames
        daily_df = pd.DataFrame.from_dict(data.daily)
        h4_df = pd.DataFrame.from_dict(data.h4)
        m15_df = pd.DataFrame.from_dict(data.m15)
        
        # The strategy expects datetime indexes
        for df in [daily_df, h4_df, m15_df]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        signal_input = {
            "daily": daily_df,
            "h4": h4_df,
            "m15": m15_df
        }
        
        # Generate the signal using our core logic
        result = strategy.generate_signal(signal_input)
        
        # Convert any numpy types to standard python types for JSON serialization
        json_compatible_result = convert_numpy_types(result)
        
        # Save the signal to the database
        signal_to_save = schemas.TradingSignalCreate(
            symbol="BTC/USDT", # Example symbol, should be dynamic
            price=m15_df['close'].iloc[-1],
            **json_compatible_result
        )
        services.create_trading_signal(db=db, signal=signal_to_save)

        return json_compatible_result

    except Exception as e:
        # If anything goes wrong, return a structured error
        raise HTTPException(status_code=500, detail=f"An error occurred during signal generation: {str(e)}") 