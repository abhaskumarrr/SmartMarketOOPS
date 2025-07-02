from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost/smartmarket"
    SECRET_KEY: str = "a_very_secret_key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # From .env
    node_env: str = "development"
    trading_mode: str = "test"
    host: str = "0.0.0.0"
    backend_port: int = 3006
    frontend_port: int = 3000
    websocket_port: int = 3001
    ml_port: int = 8000
    questdb_host: str = "localhost"
    questdb_http_port: int = 9000
    questdb_ilp_port: int = 9009
    questdb_pg_port: int = 8812
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    jwt_secret: str = "your-super-secret-jwt-key-here-change-this"
    jwt_expires_in: str = "1h"
    encryption_key: str = "your-32-byte-encryption-key-for-api-secrets"
    next_public_api_url: str = "http://localhost:3006"
    cors_origin: str = "http://localhost:3000"
    delta_exchange_api_key: Optional[str] = None
    delta_exchange_api_secret: Optional[str] = None
    delta_exchange_testnet: bool = True
    delta_exchange_base_url: str = "https://testnet-api.delta.exchange"
    ml_model_endpoint: str = "http://localhost:8000"
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    log_level: str = "info"
    enable_debug_logs: bool = False
    enable_paper_trading: bool = True
    market_data_update_interval: int = 5
    default_position_size: float = 0.1
    max_daily_trades: int = 50
    api_rate_limit: int = 100
    cache_ttl: int = 300
    max_concurrent_connections: int = 100

    class Config:
        env_file = ".env"

settings = Settings()

