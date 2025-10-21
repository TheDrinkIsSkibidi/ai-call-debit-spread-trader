from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime

from src.core.config import settings
from src.core.database import create_tables, get_db
from src.data_ingestion.market_data import MarketDataProvider, TechnicalFilter
from src.spread_constructor.spread_builder import SpreadConstructor
from src.optimizer.strategy_optimizer import StrategyOptimizer
from src.llm_integration.thesis_scorer import LLMThesisScorer
from src.ml_classifier.entry_classifier import MLEntryClassifier
from src.trade_execution.trade_manager import TradeExecutionManager
from src.trade_journal.journal_analyzer import TradeJournalAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="AI Call Debit Spread Trader",
    description="AI-assisted trading platform for call debit spreads",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components
market_provider = MarketDataProvider(settings.alpaca_api_key, settings.polygon_api_key)
technical_filter = TechnicalFilter(market_provider)
spread_constructor = SpreadConstructor(market_provider)
optimizer = StrategyOptimizer(market_provider)
llm_scorer = LLMThesisScorer()
ml_classifier = MLEntryClassifier()
trade_manager = TradeExecutionManager()
journal_analyzer = TradeJournalAnalyzer()

# Pydantic models for API requests/responses
class ScanRequest(BaseModel):
    symbols: List[str]
    max_spreads_per_symbol: int = 3

class ExecuteTradeRequest(BaseModel):
    spread_id: str
    quantity: int = 1
    force_execute: bool = False

class OptimizationRequest(BaseModel):
    symbols: List[str]
    lookback_days: int = 252
    n_trials: int = 50

class TradeResponse(BaseModel):
    success: bool
    message: str
    trade_id: Optional[int] = None
    details: Optional[Dict] = None

# Startup event
@app.on_event("startup")
async def startup_event():
    create_tables()
    print("AI Call Debit Spread Trader API started")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.1.0"
    }

# Market data endpoints
@app.get("/market/snapshot/{symbol}")
async def get_market_snapshot(symbol: str):
    """Get current market snapshot for a symbol"""
    try:
        snapshot = market_provider.get_market_snapshot(symbol)
        return {
            "symbol": snapshot.symbol,
            "price": snapshot.price,
            "volume": snapshot.volume,
            "rsi": snapshot.rsi,
            "ema_8": snapshot.ema_8,
            "ema_21": snapshot.ema_21,
            "iv_rank": snapshot.iv_rank,
            "timestamp": snapshot.timestamp.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/candidates")
async def get_candidate_symbols():
    """Get candidate symbols that pass technical filters"""
    try:
        # Default universe - can be made configurable
        universe = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        candidates = technical_filter.generate_candidates(universe)
        return {
            "candidates": candidates,
            "total_count": len(candidates),
            "universe_size": len(universe),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Spread analysis endpoints
@app.post("/spreads/scan")
async def scan_spreads(request: ScanRequest):
    """Scan for viable call debit spreads"""
    try:
        all_spreads = []
        market_snapshots = {}
        
        for symbol in request.symbols:
            # Get market data
            market_data = market_provider.get_market_snapshot(symbol)
            market_snapshots[symbol] = market_data
            
            # Get spreads for symbol
            spreads = spread_constructor.construct_spreads(symbol)
            
            # Limit spreads per symbol
            spreads = spreads[:request.max_spreads_per_symbol]
            
            all_spreads.extend(spreads)
        
        # Score spreads with AI
        scored_spreads = llm_scorer.batch_score_spreads(all_spreads, market_snapshots)
        
        # Get ML predictions
        ml_predictions = []
        for spread, llm_score in scored_spreads:
            market_data = market_snapshots[spread.symbol]
            ml_pred = ml_classifier.predict_entry(spread, market_data)
            ml_predictions.append(ml_pred)
        
        # Format response
        results = []
        for i, (spread, llm_score) in enumerate(scored_spreads):
            ml_pred = ml_predictions[i] if i < len(ml_predictions) else None
            
            result = {
                "spread": spread.to_dict(),
                "llm_analysis": {
                    "confidence_score": llm_score.confidence_score,
                    "reasoning": llm_score.reasoning[:200],  # Truncate for API
                    "recommendation": llm_score.recommendation
                },
                "ml_prediction": {
                    "win_probability": ml_pred.win_probability if ml_pred else 0.5,
                    "confidence": ml_pred.confidence if ml_pred else 0.0
                },
                "market_context": {
                    "current_price": market_snapshots[spread.symbol].price,
                    "rsi": market_snapshots[spread.symbol].rsi,
                    "iv_rank": market_snapshots[spread.symbol].iv_rank
                }
            }
            results.append(result)
        
        return {
            "spreads": results,
            "total_count": len(results),
            "scan_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/spreads/optimize")
async def optimize_strategy(
    symbols: str = "SPY,QQQ,AAPL",
    lookback_days: int = 252,
    n_trials: int = 50
):
    """Optimize strategy parameters using historical data"""
    try:
        symbol_list = symbols.split(",")
        
        optimization_result = optimizer.optimize_strategy(
            symbol_list, lookback_days, n_trials
        )
        
        return {
            "best_parameters": optimization_result.best_params,
            "best_expectancy": optimization_result.best_value,
            "backtest_results": optimization_result.backtest_results,
            "optimization_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Trading endpoints
@app.post("/trades/execute")
async def execute_trade(request: ExecuteTradeRequest):
    """Execute a specific trade"""
    try:
        # This would need to map spread_id to actual spread object
        # For now, return a placeholder response
        return TradeResponse(
            success=False,
            message="Trade execution not implemented in demo",
            details={"spread_id": request.spread_id, "quantity": request.quantity}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/positions")
async def get_positions():
    """Get current open positions"""
    try:
        account_status = trade_manager.get_account_status()
        
        # Get position updates
        position_updates = trade_manager.monitor_positions()
        
        return {
            "account_status": account_status,
            "active_positions": len(trade_manager.active_positions),
            "position_updates": position_updates,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trades/close/{trade_id}")
async def close_position(trade_id: int, reason: str = "manual"):
    """Manually close a position"""
    try:
        result = trade_manager.force_close_position(trade_id, reason)
        
        return TradeResponse(
            success=result.success,
            message=result.error_message or "Position closed successfully",
            trade_id=result.trade_id,
            details={
                "order_ids": result.order_ids,
                "filled_price": result.filled_price,
                "execution_timestamp": result.execution_timestamp.isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/analytics/performance")
async def get_performance_analytics(days_back: int = 30):
    """Get performance analytics"""
    try:
        trades_df = journal_analyzer.get_all_trades(days_back)
        
        if trades_df.empty:
            return {
                "message": "No trades found for the specified period",
                "days_back": days_back
            }
        
        performance_metrics = journal_analyzer.calculate_performance_metrics(trades_df)
        monthly_performance = journal_analyzer.analyze_monthly_performance(trades_df)
        trade_distribution = journal_analyzer.analyze_trade_distribution(trades_df)
        ai_performance = journal_analyzer.analyze_ai_model_performance(trades_df)
        
        return {
            "performance_metrics": performance_metrics.__dict__,
            "monthly_performance": monthly_performance.to_dict('records') if not monthly_performance.empty else [],
            "trade_distribution": trade_distribution,
            "ai_model_performance": ai_performance,
            "analysis_period": f"Last {days_back} days",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/insights")
async def get_ai_insights():
    """Generate AI-powered trading insights"""
    try:
        trades_df = journal_analyzer.get_all_trades()
        insights = journal_analyzer.generate_ai_insights(trades_df)
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ML Model endpoints
@app.post("/ml/train")
async def train_ml_model(background_tasks: BackgroundTasks):
    """Train the ML entry classifier"""
    try:
        def train_model():
            trades_df = journal_analyzer.get_all_trades()
            if not trades_df.empty:
                trade_history = trades_df.to_dict('records')
                metrics = ml_classifier.train_model(trade_history)
                ml_classifier.save_model()
                print(f"Model training completed. Accuracy: {metrics.accuracy:.3f}")
        
        background_tasks.add_task(train_model)
        
        return {
            "message": "Model training started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/diagnostics")
async def get_ml_diagnostics():
    """Get ML model diagnostics"""
    try:
        diagnostics = ml_classifier.model_diagnostics()
        return diagnostics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/config/status")
async def get_config_status():
    """Get current configuration status"""
    return {
        "environment": settings.environment,
        "debug": settings.debug,
        "paper_trading": settings.paper_trading,
        "max_positions": settings.max_positions,
        "max_risk_per_trade": settings.max_risk_per_trade,
        "api_keys_configured": {
            "openai": bool(settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here"),
            "anthropic": bool(settings.anthropic_api_key and settings.anthropic_api_key != "your_anthropic_api_key_here"),
            "alpaca": bool(settings.alpaca_api_key and settings.alpaca_api_key != "your_alpaca_api_key_here")
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )