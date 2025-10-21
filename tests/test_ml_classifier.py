import pytest
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml_classifier.entry_classifier import MLEntryClassifier, MLPrediction
from src.data_ingestion.market_data import MarketSnapshot

class TestMLEntryClassifier:
    
    @pytest.fixture
    def ml_classifier(self):
        """Create ML classifier instance"""
        return MLEntryClassifier()
    
    @pytest.fixture
    def sample_trade_history(self):
        """Create sample trade history for training"""
        return [
            {
                'symbol': 'SPY',
                'entry_date': '2024-01-15',
                'exit_date': '2024-02-15',
                'long_strike': 450.0,
                'short_strike': 455.0,
                'entry_debit': 1.5,
                'exit_credit': 3.0,
                'pnl': 150.0,
                'roi': 1.0,  # 100% return
                'entry_price': 448.0,
                'market_data': '{"rsi": 65, "iv_rank": 45}'
            },
            {
                'symbol': 'QQQ',
                'entry_date': '2024-01-20',
                'exit_date': '2024-02-20',
                'long_strike': 380.0,
                'short_strike': 385.0,
                'entry_debit': 2.0,
                'exit_credit': 0.5,
                'pnl': -150.0,
                'roi': -0.75,  # -75% return
                'entry_price': 378.0,
                'market_data': '{"rsi": 35, "iv_rank": 75}'
            }
        ] * 30  # Repeat to get enough samples
    
    @pytest.fixture
    def sample_spread(self):
        """Create sample spread for prediction"""
        from types import SimpleNamespace
        
        spread = SimpleNamespace()
        spread.symbol = "SPY"
        spread.long_strike = 450.0
        spread.short_strike = 455.0
        spread.net_debit = 1.5
        spread.max_profit = 3.5
        spread.max_loss = 1.5
        spread.breakeven = 451.5
        spread.risk_reward_ratio = 2.33
        spread.days_to_expiration = 30
        
        # Mock option objects
        spread.long_option = SimpleNamespace()
        spread.long_option.delta = 0.5
        spread.long_option.implied_volatility = 0.25
        
        spread.short_option = SimpleNamespace()
        spread.short_option.delta = 0.3
        spread.short_option.implied_volatility = 0.23
        
        return spread
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        return MarketSnapshot(
            symbol="SPY",
            price=450.0,
            volume=10000000,
            timestamp=pd.Timestamp.now(),
            rsi=65.0,
            ema_8=448.0,
            ema_21=445.0,
            iv_rank=45.0
        )
    
    def test_prepare_features(self, ml_classifier, sample_spread, sample_market_data):
        """Test feature preparation"""
        features = ml_classifier.prepare_features(sample_spread, sample_market_data)
        
        assert features.shape == (1, len(ml_classifier.feature_columns))
        assert isinstance(features, np.ndarray)
        
        # Check that features are numeric
        assert not np.isnan(features).any()
        assert np.isfinite(features).all()
    
    def test_train_model(self, ml_classifier, sample_trade_history):
        """Test model training"""
        metrics = ml_classifier.train_model(sample_trade_history)
        
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'precision')
        assert hasattr(metrics, 'recall')
        assert hasattr(metrics, 'f1_score')
        assert hasattr(metrics, 'auc_roc')
        
        # Check that model was trained
        assert ml_classifier.model is not None
        assert ml_classifier.trained_date is not None
    
    def test_predict_entry(self, ml_classifier, sample_trade_history, sample_spread, sample_market_data):
        """Test entry prediction"""
        # First train the model
        ml_classifier.train_model(sample_trade_history)
        
        # Make prediction
        prediction = ml_classifier.predict_entry(sample_spread, sample_market_data)
        
        assert isinstance(prediction, MLPrediction)
        assert 0 <= prediction.win_probability <= 1
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.feature_importance, dict)
        assert prediction.model_version is not None
    
    def test_batch_predict(self, ml_classifier, sample_trade_history, sample_spread, sample_market_data):
        """Test batch prediction"""
        # Train model
        ml_classifier.train_model(sample_trade_history)
        
        # Prepare batch data
        batch_data = [(sample_spread, sample_market_data)] * 3
        
        predictions = ml_classifier.batch_predict(batch_data)
        
        assert len(predictions) == 3
        assert all(isinstance(pred, MLPrediction) for pred in predictions)
    
    def test_feature_importance(self, ml_classifier, sample_trade_history):
        """Test feature importance extraction"""
        # Train model
        ml_classifier.train_model(sample_trade_history)
        
        importance = ml_classifier.get_feature_importance(top_n=5)
        
        assert isinstance(importance, dict)
        assert len(importance) <= 5
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_model_diagnostics(self, ml_classifier, sample_trade_history):
        """Test model diagnostics"""
        # Test before training
        diagnostics = ml_classifier.model_diagnostics()
        assert "error" in diagnostics or "model_version" in diagnostics
        
        # Train model
        ml_classifier.train_model(sample_trade_history)
        
        # Test after training
        diagnostics = ml_classifier.model_diagnostics()
        assert "model_version" in diagnostics
        assert "trained_date" in diagnostics
        assert "feature_count" in diagnostics
    
    def test_save_load_model(self, ml_classifier, sample_trade_history, tmp_path):
        """Test model saving and loading"""
        # Train model
        ml_classifier.train_model(sample_trade_history)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        ml_classifier.save_model(str(model_path))
        
        # Create new classifier and load model
        new_classifier = MLEntryClassifier()
        new_classifier.load_model(str(model_path))
        
        assert new_classifier.model is not None
        assert new_classifier.trained_date is not None
        assert new_classifier.feature_names == ml_classifier.feature_names

if __name__ == "__main__":
    pytest.main([__file__, "-v"])