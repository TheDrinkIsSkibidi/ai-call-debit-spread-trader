import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    anthropic_api_key: str
    alpaca_api_key: str
    alpaca_secret_key: str
    polygon_api_key: Optional[str] = None
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite:///./ai_trader.db"
    postgres_url: Optional[str] = None
    
    # Trading Configuration
    max_risk_per_trade: float = 0.02
    max_positions: int = 5
    paper_trading: bool = True
    
    # Security
    secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Monitoring
    prometheus_port: int = 8000
    
    class Config:
        env_file = ".env"


settings = Settings()