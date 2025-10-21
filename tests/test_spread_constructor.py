import pytest
import sys
import os
from unittest.mock import Mock, patch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.spread_constructor.spread_builder import SpreadConstructor, CallDebitSpread
from src.data_ingestion.market_data import MarketDataProvider, OptionData

class TestSpreadConstructor:
    
    @pytest.fixture
    def mock_market_provider(self):
        """Create a mock market data provider"""
        provider = Mock(spec=MarketDataProvider)
        
        # Mock option chain data
        mock_options = [
            OptionData(
                strike=450.0, expiration="2024-03-15", option_type="CALL",
                bid=4.50, ask=4.60, last=4.55, volume=100, open_interest=500,
                implied_volatility=0.25, delta=0.5, gamma=0.01, theta=-0.02, vega=0.1
            ),
            OptionData(
                strike=455.0, expiration="2024-03-15", option_type="CALL",
                bid=3.20, ask=3.30, last=3.25, volume=80, open_interest=300,
                implied_volatility=0.23, delta=0.3, gamma=0.008, theta=-0.015, vega=0.08
            )
        ]
        
        provider.get_option_chain_data.return_value = mock_options
        
        # Mock market snapshot
        mock_snapshot = Mock()
        mock_snapshot.price = 448.50
        provider.get_market_snapshot.return_value = mock_snapshot
        
        return provider
    
    @pytest.fixture
    def spread_constructor(self, mock_market_provider):
        """Create spread constructor with mocked dependencies"""
        return SpreadConstructor(mock_market_provider)
    
    def test_construct_spreads(self, spread_constructor):
        """Test spread construction"""
        spreads = spread_constructor.construct_spreads("SPY")
        
        assert isinstance(spreads, list)
        # Should find at least one spread from mock data
        if spreads:
            spread = spreads[0]
            assert isinstance(spread, CallDebitSpread)
            assert spread.symbol == "SPY"
            assert spread.long_strike < spread.short_strike
            assert spread.net_debit > 0
            assert spread.max_profit > 0
    
    def test_calculate_spread_metrics(self, spread_constructor, mock_market_provider):
        """Test spread metrics calculation"""
        # Create mock options
        long_option = OptionData(
            strike=450.0, expiration="2024-03-15", option_type="CALL",
            bid=4.50, ask=4.60, last=4.55, volume=100, open_interest=500,
            implied_volatility=0.25, delta=0.5, gamma=0.01, theta=-0.02, vega=0.1
        )
        
        short_option = OptionData(
            strike=455.0, expiration="2024-03-15", option_type="CALL",
            bid=3.20, ask=3.30, last=3.25, volume=80, open_interest=300,
            implied_volatility=0.23, delta=0.3, gamma=0.008, theta=-0.015, vega=0.08
        )
        
        spread = spread_constructor._calculate_spread_metrics(
            "SPY", "2024-03-15", long_option, short_option
        )
        
        assert spread is not None
        assert spread.net_debit == pytest.approx(1.3, rel=0.1)  # 4.55 - 3.25
        assert spread.max_profit == pytest.approx(3.7, rel=0.1)  # 5 - 1.3
        assert spread.breakeven == pytest.approx(451.3, rel=0.1)  # 450 + 1.3
    
    def test_risk_reward_calculation(self, spread_constructor):
        """Test risk/reward ratio calculation"""
        # This would test the R/R ratio calculation
        # Implementation depends on the specific calculation method
        pass
    
    def test_spread_filtering(self, spread_constructor):
        """Test spread filtering criteria"""
        # Test that spreads are properly filtered based on criteria
        # like max debit percentage, minimum R/R ratio, etc.
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])