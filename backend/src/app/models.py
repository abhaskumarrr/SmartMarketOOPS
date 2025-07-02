from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Float, JSON
from sqlalchemy.orm import relationship
from .database import Base
import uuid
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    is_verified = Column(Boolean, default=False)
    verification_token = Column(String, nullable=True)
    verification_token_expiry = Column(DateTime, nullable=True)
    reset_token = Column(String, nullable=True)
    reset_token_expiry = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime, nullable=True)
    oauth_provider = Column(String, nullable=True)
    oauth_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    api_keys = relationship("ApiKey", back_populates="user")
    orders = relationship("Order", back_populates="user")

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key = Column(String, unique=True, index=True)
    encrypted_data = Column(String)
    user_id = Column(String, ForeignKey("users.id"))
    name = Column(String, default="Default")
    scopes = Column(String)
    expiry = Column(DateTime)
    environment = Column(String, default="testnet")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    is_revoked = Column(Boolean, default=False)

    user = relationship("User", back_populates="api_keys")

class TradingSignal(Base):
    __tablename__ = "trading_signals"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, index=True)
    type = Column(String)
    direction = Column(String)
    strength = Column(String)
    timeframe = Column(String)
    price = Column(Float)
    target_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    confidence_score = Column(Integer)
    expected_return = Column(Float)
    expected_risk = Column(Float)
    risk_reward_ratio = Column(Float)
    generated_at = Column(DateTime)
    expires_at = Column(DateTime, nullable=True)
    source = Column(String)
    signal_signal_metadata = Column(JSON, nullable=True)
    prediction_values = Column(JSON)
    validated_at = Column(DateTime, nullable=True)
    validation_status = Column(Boolean, default=False)
    validation_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    signal_id = Column(String, ForeignKey("trading_signals.id"), nullable=True)
    exchange_order_id = Column(String, nullable=True)
    symbol = Column(String)
    type = Column(String)
    side = Column(String)
    price = Column(Float)
    amount = Column(Float)
    status = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="orders")
    signal = relationship("TradingSignal")
