from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

# --- User Schemas ---
class UserBase(BaseModel):
    email: str
    name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: str
    role: str
    is_verified: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# --- API Key Schemas ---
class ApiKeyBase(BaseModel):
    name: str
    environment: str = "testnet"

class ApiKeyCreate(ApiKeyBase):
    pass

class ApiKey(ApiKeyBase):
    id: str
    key: str
    expiry: datetime
    is_revoked: bool
    last_used_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True

# --- Trading Signal Schemas ---
class TradingSignalBase(BaseModel):
    symbol: str
    type: str
    direction: str
    strength: float
    confidence: float
    reason: str
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None

class TradingSignalCreate(TradingSignalBase):
    price: float
    signal_metadata: Optional[dict] = None

class TradingSignal(TradingSignalBase):
    id: str
    price: float
    signal_metadata: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True

# --- Order Schemas ---
class OrderBase(BaseModel):
    symbol: str
    side: str
    type: str
    quantity: float
    price: Optional[float] = None

class OrderCreate(OrderBase):
    pass

class Order(OrderBase):
    id: str
    status: str
    filled_quantity: Optional[float] = None
    filled_price: Optional[float] = None
    fee: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True

