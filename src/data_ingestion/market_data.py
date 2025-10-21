import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/optlib'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import requests
from dataclasses import dataclass

from optlib.api import get_option_chain, get_stock_price
from optlib.gbs import black_scholes


@dataclass
class OptionData:
    strike: float
    expiration: str
    option_type: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    rsi: Optional[float] = None
    ema_8: Optional[float] = None
    ema_21: Optional[float] = None
    iv_rank: Optional[float] = None


class MarketDataProvider:
    def __init__(self, alpaca_key: str = None, polygon_key: str = None):
        self.alpaca_key = alpaca_key
        self.polygon_key = polygon_key
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_option_chain_data(self, symbol: str, expiration: str = None) -> List[OptionData]:
        try:
            # Use optlib to get option chain
            chain_data = get_option_chain(symbol, expiration)
            
            options = []
            for option in chain_data:
                # Calculate Greeks using optlib's Black-Scholes implementation
                spot_price = get_stock_price(symbol)
                
                greeks = black_scholes(
                    S=spot_price,
                    K=option['strike'],
                    T=self._time_to_expiration(option['expiration']),
                    r=0.05,  # Risk-free rate
                    sigma=option.get('implied_volatility', 0.2),
                    option_type=option['type']
                )
                
                options.append(OptionData(
                    strike=option['strike'],
                    expiration=option['expiration'],
                    option_type=option['type'],
                    bid=option.get('bid', 0),
                    ask=option.get('ask', 0),
                    last=option.get('last', 0),
                    volume=option.get('volume', 0),
                    open_interest=option.get('open_interest', 0),
                    implied_volatility=option.get('implied_volatility', 0),
                    delta=greeks.get('delta', 0),
                    gamma=greeks.get('gamma', 0),
                    theta=greeks.get('theta', 0),
                    vega=greeks.get('vega', 0)
                ))
            
            return options
            
        except Exception as e:
            print(f"Error fetching option chain for {symbol}: {e}")
            return []
    
    def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            # Get historical data for technical indicators
            hist_data = self.get_stock_data(symbol, period="50d")
            
            if not hist_data.empty:
                # Calculate technical indicators
                rsi = self._calculate_rsi(hist_data['Close'])
                ema_8 = self._calculate_ema(hist_data['Close'], 8)
                ema_21 = self._calculate_ema(hist_data['Close'], 21)
                
                # Calculate IV rank (simplified)
                iv_rank = self._calculate_iv_rank(symbol)
                
                return MarketSnapshot(
                    symbol=symbol,
                    price=current_price,
                    volume=hist_data['Volume'].iloc[-1] if not hist_data.empty else 0,
                    timestamp=datetime.now(),
                    rsi=rsi,
                    ema_8=ema_8,
                    ema_21=ema_21,
                    iv_rank=iv_rank
                )
            else:
                return MarketSnapshot(
                    symbol=symbol,
                    price=current_price,
                    volume=0,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            print(f"Error getting market snapshot for {symbol}: {e}")
            return MarketSnapshot(symbol, 0, 0, datetime.now())
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> float:
        ema = prices.ewm(span=period).mean()
        return ema.iloc[-1] if not ema.empty else prices.iloc[-1]
    
    def _calculate_iv_rank(self, symbol: str) -> float:
        # Simplified IV rank calculation
        # In production, this would use historical IV data
        try:
            # Get 1-year of data for IV calculation
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if not hist.empty:
                # Calculate historical volatility as proxy for IV
                returns = np.log(hist['Close'] / hist['Close'].shift(1))
                current_vol = returns.rolling(window=30).std().iloc[-1] * np.sqrt(252)
                hist_vols = returns.rolling(window=30).std() * np.sqrt(252)
                
                # IV rank = percentile of current vol in historical distribution
                iv_rank = (hist_vols < current_vol).sum() / len(hist_vols.dropna()) * 100
                return iv_rank
            
        except Exception as e:
            print(f"Error calculating IV rank for {symbol}: {e}")
        
        return 50.0  # Default neutral IV rank
    
    def _time_to_expiration(self, expiration_str: str) -> float:
        try:
            exp_date = datetime.strptime(expiration_str, "%Y-%m-%d")
            days_to_exp = (exp_date - datetime.now()).days
            return max(days_to_exp / 365.0, 0.001)  # Minimum 0.001 years
        except:
            return 0.1  # Default 0.1 years


class TechnicalFilter:
    def __init__(self, market_provider: MarketDataProvider):
        self.market_provider = market_provider
    
    def apply_market_filter(self, spy_data: MarketSnapshot) -> bool:
        """Apply market-wide filters: SPY > 20EMA, RSI>55, VIX<20"""
        if not spy_data.ema_21 or not spy_data.rsi:
            return False
            
        # Check if SPY is above 20EMA (using 21EMA as proxy)
        above_ema = spy_data.price > spy_data.ema_21
        
        # Check RSI > 55
        rsi_ok = spy_data.rsi > 55
        
        # TODO: Add VIX check when available
        vix_ok = True  # Placeholder
        
        return above_ema and rsi_ok and vix_ok
    
    def apply_ticker_filter(self, ticker_data: MarketSnapshot) -> bool:
        """Apply ticker-specific filters: 8EMA > 21EMA, RSI>55, volume > avg10"""
        if not all([ticker_data.ema_8, ticker_data.ema_21, ticker_data.rsi]):
            return False
        
        # Check if 8EMA > 21EMA (uptrend)
        uptrend = ticker_data.ema_8 > ticker_data.ema_21
        
        # Check RSI > 55
        rsi_ok = ticker_data.rsi > 55
        
        # TODO: Add volume comparison to 10-day average
        volume_ok = True  # Placeholder
        
        return uptrend and rsi_ok and volume_ok
    
    def generate_candidates(self, universe: List[str]) -> List[str]:
        """Generate list of candidate tickers that pass all filters"""
        # First check market filter with SPY
        spy_data = self.market_provider.get_market_snapshot("SPY")
        if not self.apply_market_filter(spy_data):
            print("Market filter failed - no candidates generated")
            return []
        
        candidates = []
        for symbol in universe:
            ticker_data = self.market_provider.get_market_snapshot(symbol)
            if self.apply_ticker_filter(ticker_data):
                candidates.append(symbol)
        
        return candidates