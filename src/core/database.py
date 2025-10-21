from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .config import settings

engine = create_engine(settings.database_url, echo=settings.debug)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    strategy_type = Column(String, default="call_debit_spread")
    
    # Spread details
    long_strike = Column(Float)
    short_strike = Column(Float)
    expiration_date = Column(DateTime)
    quantity = Column(Integer)
    
    # Entry details
    entry_date = Column(DateTime, default=datetime.utcnow)
    entry_debit = Column(Float)
    entry_price = Column(Float)
    
    # Exit details
    exit_date = Column(DateTime, nullable=True)
    exit_credit = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    
    # Performance
    pnl = Column(Float, nullable=True)
    roi = Column(Float, nullable=True)
    
    # AI Scores
    llm_score = Column(Float, nullable=True)
    llm_reasoning = Column(String, nullable=True)
    ml_probability = Column(Float, nullable=True)
    
    # Market data
    market_data = Column(JSON)
    
    # Status
    status = Column(String, default="open")  # open, closed, expired
    is_paper_trade = Column(Boolean, default=True)


class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Price data
    price = Column(Float)
    volume = Column(Integer)
    
    # Technical indicators
    rsi = Column(Float, nullable=True)
    ema_8 = Column(Float, nullable=True)
    ema_21 = Column(Float, nullable=True)
    iv_rank = Column(Float, nullable=True)
    
    # Option chain data
    option_chain = Column(JSON, nullable=True)


class AIInsight(Base):
    __tablename__ = "ai_insights"
    
    id = Column(Integer, primary_key=True, index=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Analysis period
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    
    # Insights
    patterns_identified = Column(JSON)
    recommendations = Column(JSON)
    confidence_adjustments = Column(JSON)
    
    # Performance metrics
    total_trades = Column(Integer)
    win_rate = Column(Float)
    avg_roi = Column(Float)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()