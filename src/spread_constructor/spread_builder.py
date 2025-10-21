import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/optlib'))

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from src.data_ingestion.market_data import MarketDataProvider, OptionData


@dataclass
class CallDebitSpread:
    symbol: str
    expiration: str
    long_strike: float
    short_strike: float
    long_option: OptionData
    short_option: OptionData
    net_debit: float
    max_profit: float
    max_loss: float
    breakeven: float
    profit_probability: float
    risk_reward_ratio: float
    days_to_expiration: int
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'expiration': self.expiration,
            'long_strike': self.long_strike,
            'short_strike': self.short_strike,
            'net_debit': self.net_debit,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'breakeven': self.breakeven,
            'profit_probability': self.profit_probability,
            'risk_reward_ratio': self.risk_reward_ratio,
            'days_to_expiration': self.days_to_expiration,
            'long_delta': self.long_option.delta,
            'short_delta': self.short_option.delta,
            'net_delta': self.long_option.delta - self.short_option.delta,
            'long_iv': self.long_option.implied_volatility,
            'short_iv': self.short_option.implied_volatility
        }


class SpreadConstructor:
    def __init__(self, market_provider: MarketDataProvider):
        self.market_provider = market_provider
        
        # Default parameters for spread construction
        self.target_long_delta = 0.5
        self.target_short_delta = 0.3
        self.min_days_to_expiration = 20
        self.max_days_to_expiration = 60
        self.min_width = 5.0  # Minimum strike width
        self.max_debit_percent = 0.6  # Max debit as % of spread width
    
    def construct_spreads(self, symbol: str) -> List[CallDebitSpread]:
        """Construct all viable call debit spreads for a symbol"""
        # Get option chain data
        option_chain = self.market_provider.get_option_chain_data(symbol)
        
        if not option_chain:
            return []
        
        # Filter options by expiration and type
        call_options = self._filter_call_options(option_chain)
        
        # Group by expiration
        expirations = self._group_by_expiration(call_options)
        
        spreads = []
        for expiration, options in expirations.items():
            spread_candidates = self._build_spreads_for_expiration(symbol, expiration, options)
            spreads.extend(spread_candidates)
        
        # Filter and rank spreads
        viable_spreads = self._filter_viable_spreads(spreads)
        return self._rank_spreads(viable_spreads)
    
    def _filter_call_options(self, option_chain: List[OptionData]) -> List[OptionData]:
        """Filter for call options within target expiration range"""
        calls = []
        current_date = datetime.now()
        
        for option in option_chain:
            if option.option_type.upper() != 'CALL':
                continue
            
            # Calculate days to expiration
            exp_date = datetime.strptime(option.expiration, "%Y-%m-%d")
            days_to_exp = (exp_date - current_date).days
            
            if self.min_days_to_expiration <= days_to_exp <= self.max_days_to_expiration:
                calls.append(option)
        
        return calls
    
    def _group_by_expiration(self, options: List[OptionData]) -> Dict[str, List[OptionData]]:
        """Group options by expiration date"""
        expirations = {}
        for option in options:
            if option.expiration not in expirations:
                expirations[option.expiration] = []
            expirations[option.expiration].append(option)
        
        return expirations
    
    def _build_spreads_for_expiration(self, symbol: str, expiration: str, options: List[OptionData]) -> List[CallDebitSpread]:
        """Build all viable spreads for a given expiration"""
        spreads = []
        
        # Sort options by strike
        options.sort(key=lambda x: x.strike)
        
        for i, long_option in enumerate(options):
            # Skip if long delta is not close to target
            if abs(long_option.delta - self.target_long_delta) > 0.15:
                continue
            
            for j, short_option in enumerate(options[i+1:], i+1):
                # Skip if short delta is not close to target
                if abs(short_option.delta - self.target_short_delta) > 0.15:
                    continue
                
                # Check minimum width
                strike_width = short_option.strike - long_option.strike
                if strike_width < self.min_width:
                    continue
                
                # Calculate spread metrics
                spread = self._calculate_spread_metrics(symbol, expiration, long_option, short_option)
                
                if spread and self._is_viable_spread(spread):
                    spreads.append(spread)
        
        return spreads
    
    def _calculate_spread_metrics(self, symbol: str, expiration: str, 
                                 long_option: OptionData, short_option: OptionData) -> Optional[CallDebitSpread]:
        """Calculate all metrics for a call debit spread"""
        try:
            # Calculate net debit (using mid prices)
            long_price = (long_option.bid + long_option.ask) / 2 if long_option.ask > 0 else long_option.last
            short_price = (short_option.bid + short_option.ask) / 2 if short_option.ask > 0 else short_option.last
            
            if long_price <= 0 or short_price <= 0:
                return None
            
            net_debit = long_price - short_price
            strike_width = short_option.strike - long_option.strike
            
            # Calculate max profit/loss
            max_profit = strike_width - net_debit
            max_loss = net_debit
            
            # Calculate breakeven
            breakeven = long_option.strike + net_debit
            
            # Calculate risk/reward ratio
            risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0
            
            # Calculate days to expiration
            exp_date = datetime.strptime(expiration, "%Y-%m-%d")
            days_to_exp = (exp_date - datetime.now()).days
            
            # Estimate profit probability (simplified)
            current_price = self.market_provider.get_market_snapshot(symbol).price
            profit_probability = self._estimate_profit_probability(current_price, breakeven, long_option.delta)
            
            return CallDebitSpread(
                symbol=symbol,
                expiration=expiration,
                long_strike=long_option.strike,
                short_strike=short_option.strike,
                long_option=long_option,
                short_option=short_option,
                net_debit=net_debit,
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven=breakeven,
                profit_probability=profit_probability,
                risk_reward_ratio=risk_reward_ratio,
                days_to_expiration=days_to_exp
            )
            
        except Exception as e:
            print(f"Error calculating spread metrics: {e}")
            return None
    
    def _estimate_profit_probability(self, current_price: float, breakeven: float, delta: float) -> float:
        """Estimate probability of profit using simplified model"""
        # This is a simplified calculation
        # In production, use more sophisticated models or option pricing models
        
        if current_price <= 0 or breakeven <= 0:
            return 0.0
        
        # Use delta as proxy for probability (simplified)
        # Delta approximates the probability of finishing ITM
        move_needed = (breakeven - current_price) / current_price
        
        # Adjust probability based on required move
        if move_needed <= 0:
            return 0.9  # Already above breakeven
        elif move_needed > 0.2:
            return 0.1  # Need large move
        else:
            # Linear interpolation between 0.9 and 0.1
            return 0.9 - (move_needed / 0.2) * 0.8
    
    def _is_viable_spread(self, spread: CallDebitSpread) -> bool:
        """Check if spread meets viability criteria"""
        # Check debit percentage of spread width
        strike_width = spread.short_strike - spread.long_strike
        debit_percentage = spread.net_debit / strike_width
        
        if debit_percentage > self.max_debit_percent:
            return False
        
        # Check minimum risk/reward ratio
        if spread.risk_reward_ratio < 0.5:
            return False
        
        # Check minimum profit probability
        if spread.profit_probability < 0.3:
            return False
        
        # Check for reasonable liquidity (simplified)
        if (spread.long_option.volume + spread.short_option.volume) < 10:
            return False
        
        return True
    
    def _filter_viable_spreads(self, spreads: List[CallDebitSpread]) -> List[CallDebitSpread]:
        """Apply additional filters to spreads"""
        viable = []
        
        for spread in spreads:
            # Additional viability checks can be added here
            if spread.net_debit > 0 and spread.max_profit > 0:
                viable.append(spread)
        
        return viable
    
    def _rank_spreads(self, spreads: List[CallDebitSpread]) -> List[CallDebitSpread]:
        """Rank spreads by attractiveness score"""
        def score_spread(spread: CallDebitSpread) -> float:
            # Composite score based on multiple factors
            score = 0.0
            
            # Risk/reward ratio (weight: 30%)
            score += spread.risk_reward_ratio * 0.3
            
            # Profit probability (weight: 40%)
            score += spread.profit_probability * 0.4
            
            # Days to expiration bonus for 30-45 days (weight: 20%)
            if 30 <= spread.days_to_expiration <= 45:
                score += 1.0 * 0.2
            else:
                score += (1.0 - abs(37.5 - spread.days_to_expiration) / 37.5) * 0.2
            
            # Delta proximity bonus (weight: 10%)
            long_delta_score = 1.0 - abs(spread.long_option.delta - self.target_long_delta) / 0.5
            short_delta_score = 1.0 - abs(spread.short_option.delta - self.target_short_delta) / 0.3
            score += (long_delta_score + short_delta_score) / 2 * 0.1
            
            return score
        
        # Sort by score (highest first)
        spreads.sort(key=score_spread, reverse=True)
        return spreads
    
    def export_spreads(self, spreads: List[CallDebitSpread], filename: str = "spreads.json"):
        """Export spreads to JSON file"""
        spreads_data = [spread.to_dict() for spread in spreads]
        
        with open(filename, 'w') as f:
            json.dump(spreads_data, f, indent=2)
        
        print(f"Exported {len(spreads)} spreads to {filename}")
    
    def get_best_spreads(self, symbols: List[str], max_per_symbol: int = 3) -> List[CallDebitSpread]:
        """Get best spreads across multiple symbols"""
        all_spreads = []
        
        for symbol in symbols:
            spreads = self.construct_spreads(symbol)
            # Take top spreads per symbol
            all_spreads.extend(spreads[:max_per_symbol])
        
        # Re-rank all spreads together
        return self._rank_spreads(all_spreads)