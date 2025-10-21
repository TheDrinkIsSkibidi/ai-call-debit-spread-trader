#!/usr/bin/env python3
"""
AI Call Debit Spread Trader - Main Entry Point

This script provides a unified entry point for running different components
of the AI trading system.
"""

import argparse
import sys
import os
import subprocess

def run_api():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend...")
    cmd = ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    subprocess.run(cmd)

def run_dashboard():
    """Start the Streamlit dashboard"""
    print("📊 Starting Streamlit dashboard...")
    cmd = ["streamlit", "run", "src/dashboard_ui/app.py", "--server.port", "8501"]
    subprocess.run(cmd)

def run_scanner():
    """Run a one-time market scan"""
    print("🔍 Running market scanner...")
    from src.data_ingestion.market_data import MarketDataProvider, TechnicalFilter
    from src.spread_constructor.spread_builder import SpreadConstructor
    
    # Initialize components
    market_provider = MarketDataProvider()
    technical_filter = TechnicalFilter(market_provider)
    spread_constructor = SpreadConstructor(market_provider)
    
    # Default universe
    universe = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    
    # Get candidates
    print("Finding candidate symbols...")
    candidates = technical_filter.generate_candidates(universe)
    print(f"Found {len(candidates)} candidates: {candidates}")
    
    # Scan for spreads
    all_spreads = []
    for symbol in candidates[:3]:  # Limit to top 3 for demo
        print(f"Scanning {symbol}...")
        spreads = spread_constructor.construct_spreads(symbol)
        all_spreads.extend(spreads[:2])  # Top 2 per symbol
    
    # Display results
    print(f"\n📈 Found {len(all_spreads)} viable spreads:")
    for spread in all_spreads:
        print(f"{spread.symbol}: {spread.long_strike}/{spread.short_strike} "
              f"(${spread.net_debit:.2f} debit, {spread.risk_reward_ratio:.2f} R/R)")

def run_backtest():
    """Run strategy backtesting"""
    print("📈 Running strategy backtest...")
    from src.optimizer.strategy_optimizer import StrategyOptimizer
    from src.data_ingestion.market_data import MarketDataProvider
    
    market_provider = MarketDataProvider()
    optimizer = StrategyOptimizer(market_provider)
    
    # Run optimization
    symbols = ["SPY", "QQQ", "AAPL"]
    print(f"Optimizing strategy for {symbols}...")
    
    result = optimizer.optimize_strategy(symbols, lookback_days=180, n_trials=20)
    
    print(f"\n🎯 Optimization Results:")
    print(f"Best Expectancy: {result.best_value:.3f}")
    print(f"Best Parameters: {result.best_params}")
    
    # Export results
    optimizer.export_optimization_results(result, "backtest_results.json")
    print("Results saved to backtest_results.json")

def train_ml_model():
    """Train the ML model"""
    print("🧠 Training ML model...")
    from src.ml_classifier.entry_classifier import MLEntryClassifier
    from src.trade_journal.journal_analyzer import TradeJournalAnalyzer
    
    # Get trade history
    analyzer = TradeJournalAnalyzer()
    trades_df = analyzer.get_all_trades()
    
    if trades_df.empty:
        print("❌ No trade history available for training")
        return
    
    # Train model
    classifier = MLEntryClassifier()
    trade_history = trades_df.to_dict('records')
    
    metrics = classifier.train_model(trade_history)
    
    print(f"✅ Model training complete!")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    
    # Save model
    classifier.save_model()
    print("Model saved to ml_entry_classifier.joblib")

def setup_database():
    """Initialize the database"""
    print("🗄️ Setting up database...")
    from src.core.database import create_tables
    
    create_tables()
    print("✅ Database tables created successfully")

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    cmd = ["pytest", "tests/", "-v"]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="AI Call Debit Spread Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py api              # Start FastAPI backend
  python main.py dashboard        # Start Streamlit dashboard  
  python main.py scan             # Run market scanner
  python main.py backtest         # Run strategy backtesting
  python main.py train            # Train ML model
  python main.py setup-db         # Initialize database
  python main.py test             # Run tests
        """
    )
    
    parser.add_argument(
        'command',
        choices=['api', 'dashboard', 'scan', 'backtest', 'train', 'setup-db', 'test'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Execute command
    commands = {
        'api': run_api,
        'dashboard': run_dashboard,
        'scan': run_scanner,
        'backtest': run_backtest,
        'train': train_ml_model,
        'setup-db': setup_database,
        'test': run_tests
    }
    
    try:
        commands[args.command]()
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()