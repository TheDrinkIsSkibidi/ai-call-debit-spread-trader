import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/optionlab'))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import optuna
from dataclasses import dataclass
import json
from datetime import datetime, timedelta

from optionlab import Strategy, get_options_chain
from src.spread_constructor.spread_builder import CallDebitSpread
from src.data_ingestion.market_data import MarketDataProvider


@dataclass 
class OptimizationResult:
    best_params: Dict[str, Any]
    best_value: float
    study_history: List[Dict]
    backtest_results: Dict
    payoff_diagram_data: Dict


@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    expectancy: float


class StrategyOptimizer:
    def __init__(self, market_provider: MarketDataProvider):
        self.market_provider = market_provider
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'strike_width': (5, 20),
            'dte_min': (20, 35),
            'dte_max': (35, 60),
            'target_delta_long': (0.4, 0.6),
            'target_delta_short': (0.2, 0.4),
            'max_debit_pct': (0.3, 0.7),
            'profit_target_pct': (0.5, 1.0),
            'stop_loss_pct': (0.3, 0.8),
            'iv_rank_min': (20, 40),
            'iv_rank_max': (60, 90)
        }
    
    def optimize_strategy(self, symbols: List[str], lookback_days: int = 252, 
                         n_trials: int = 100) -> OptimizationResult:
        """Optimize call debit spread strategy parameters using Optuna"""
        
        # Generate historical data for backtesting
        historical_data = self._prepare_historical_data(symbols, lookback_days)
        
        def objective(trial):
            # Sample parameters
            params = {
                'strike_width': trial.suggest_float('strike_width', *self.param_ranges['strike_width']),
                'dte_min': trial.suggest_int('dte_min', *self.param_ranges['dte_min']),
                'dte_max': trial.suggest_int('dte_max', *self.param_ranges['dte_max']),
                'target_delta_long': trial.suggest_float('target_delta_long', *self.param_ranges['target_delta_long']),
                'target_delta_short': trial.suggest_float('target_delta_short', *self.param_ranges['target_delta_short']),
                'max_debit_pct': trial.suggest_float('max_debit_pct', *self.param_ranges['max_debit_pct']),
                'profit_target_pct': trial.suggest_float('profit_target_pct', *self.param_ranges['profit_target_pct']),
                'stop_loss_pct': trial.suggest_float('stop_loss_pct', *self.param_ranges['stop_loss_pct']),
                'iv_rank_min': trial.suggest_float('iv_rank_min', *self.param_ranges['iv_rank_min']),
                'iv_rank_max': trial.suggest_float('iv_rank_max', *self.param_ranges['iv_rank_max'])
            }
            
            # Run backtest with these parameters
            backtest_results = self._run_backtest(historical_data, params)
            
            # Return objective value (we want to maximize expectancy)
            return backtest_results.expectancy
        
        # Create and run optimization study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and run final backtest
        best_params = study.best_params
        final_backtest = self._run_backtest(historical_data, best_params)
        
        # Generate payoff diagram for best strategy
        payoff_data = self._generate_payoff_diagram(best_params)
        
        return OptimizationResult(
            best_params=best_params,
            best_value=study.best_value,
            study_history=[{
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params
            } for trial in study.trials],
            backtest_results=final_backtest.__dict__,
            payoff_diagram_data=payoff_data
        )
    
    def evaluate_spread_with_optionlab(self, spread: CallDebitSpread) -> Dict:
        """Evaluate a single spread using OptionLab's Strategy class"""
        try:
            # Create OptionLab strategy
            strategy = Strategy()
            
            # Add long call
            strategy.add_option(
                option_type='call',
                strike=spread.long_strike,
                premium=spread.long_option.last,
                n=1,  # Buy 1 contract
                action='buy'
            )
            
            # Add short call
            strategy.add_option(
                option_type='call',
                strike=spread.short_strike,
                premium=spread.short_option.last,
                n=1,  # Sell 1 contract
                action='sell'
            )
            
            # Calculate strategy metrics
            current_price = self.market_provider.get_market_snapshot(spread.symbol).price
            
            # Generate price range for analysis
            price_range = np.linspace(
                current_price * 0.8,
                current_price * 1.2,
                100
            )
            
            # Calculate P&L for each price point
            pnl_values = []
            for price in price_range:
                pnl = strategy.calculate_pnl(price, spread.days_to_expiration)
                pnl_values.append(pnl)
            
            # Calculate key metrics
            max_profit = max(pnl_values)
            max_loss = min(pnl_values)
            breakeven_prices = self._find_breakeven_points(price_range, pnl_values)
            
            # Calculate probability of profit
            prob_profit = self._calculate_probability_of_profit(
                current_price, breakeven_prices, spread.long_option.implied_volatility, 
                spread.days_to_expiration
            )
            
            return {
                'max_profit': max_profit,
                'max_loss': max_loss,
                'breakeven_prices': breakeven_prices,
                'probability_of_profit': prob_profit,
                'price_range': price_range.tolist(),
                'pnl_values': pnl_values,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"Error evaluating spread with OptionLab: {e}")
            return {}
    
    def _prepare_historical_data(self, symbols: List[str], lookback_days: int) -> Dict:
        """Prepare historical market data for backtesting"""
        historical_data = {}
        
        for symbol in symbols:
            # Get historical price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            hist_data = self.market_provider.get_stock_data(symbol, period=f"{lookback_days}d")
            
            if not hist_data.empty:
                historical_data[symbol] = {
                    'prices': hist_data,
                    'start_date': start_date,
                    'end_date': end_date
                }
        
        return historical_data
    
    def _run_backtest(self, historical_data: Dict, params: Dict) -> BacktestMetrics:
        """Run backtest with given parameters"""
        all_trades = []
        
        for symbol, data in historical_data.items():
            trades = self._simulate_trades(symbol, data, params)
            all_trades.extend(trades)
        
        if not all_trades:
            return BacktestMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate metrics
        winning_trades = [t for t in all_trades if t['pnl'] > 0]
        losing_trades = [t for t in all_trades if t['pnl'] <= 0]
        
        total_trades = len(all_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0
        total_pnl = sum(t['pnl'] for t in all_trades)
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum([t['pnl'] for t in all_trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['pnl'] for t in all_trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            expectancy=expectancy
        )
    
    def _simulate_trades(self, symbol: str, data: Dict, params: Dict) -> List[Dict]:
        """Simulate trades for a symbol using given parameters"""
        trades = []
        prices_df = data['prices']
        
        # Simple simulation - enter trades weekly
        for i in range(0, len(prices_df) - 30, 7):  # Weekly entries
            entry_date = prices_df.index[i]
            entry_price = prices_df.iloc[i]['Close']
            
            # Simulate spread entry
            trade = self._simulate_spread_trade(
                symbol, entry_date, entry_price, params, prices_df[i:]
            )
            
            if trade:
                trades.append(trade)
        
        return trades
    
    def _simulate_spread_trade(self, symbol: str, entry_date, entry_price: float, 
                              params: Dict, future_prices: pd.DataFrame) -> Optional[Dict]:
        """Simulate a single spread trade"""
        try:
            # Simplified simulation - assume we can construct a spread
            # In reality, this would use historical option data
            
            # Calculate strikes based on deltas (simplified)
            strike_width = params['strike_width']
            long_strike = entry_price * 1.02  # Slightly OTM
            short_strike = long_strike + strike_width
            
            # Estimate premium and debit (simplified)
            estimated_long_premium = entry_price * 0.03  # 3% of stock price
            estimated_short_premium = entry_price * 0.02  # 2% of stock price
            net_debit = estimated_long_premium - estimated_short_premium
            
            # Check if trade meets criteria
            debit_pct = net_debit / strike_width
            if debit_pct > params['max_debit_pct']:
                return None
            
            # Simulate trade outcome
            max_profit = strike_width - net_debit
            max_loss = net_debit
            
            profit_target = net_debit + (max_profit * params['profit_target_pct'])
            stop_loss = net_debit - (max_loss * params['stop_loss_pct'])
            
            # Check exit conditions over next 30 days
            for j, (date, row) in enumerate(future_prices.head(30).iterrows()):
                if j == 0:
                    continue
                
                current_price = row['Close']
                
                # Estimate current spread value (simplified)
                if current_price >= short_strike:
                    current_value = strike_width  # Max profit
                elif current_price <= long_strike:
                    current_value = 0  # Max loss
                else:
                    # Linear interpolation (simplified)
                    current_value = (current_price - long_strike) / strike_width * strike_width
                
                pnl = current_value - net_debit
                
                # Check exit conditions
                if current_value >= profit_target or current_value <= stop_loss or j >= 29:
                    return {
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'long_strike': long_strike,
                        'short_strike': short_strike,
                        'net_debit': net_debit,
                        'pnl': pnl,
                        'days_held': j,
                        'exit_reason': 'profit_target' if current_value >= profit_target else 
                                     'stop_loss' if current_value <= stop_loss else 'expiration'
                    }
            
            return None
            
        except Exception as e:
            print(f"Error simulating trade: {e}")
            return None
    
    def _generate_payoff_diagram(self, params: Dict) -> Dict:
        """Generate payoff diagram data for optimized strategy"""
        # This would generate data for visualization
        # Simplified implementation
        return {
            'strikes': [100, 105, 110, 115, 120],
            'payoff': [-5, -3, 0, 3, 5],
            'breakeven': 103,
            'max_profit': 5,
            'max_loss': -5
        }
    
    def _find_breakeven_points(self, price_range: np.ndarray, pnl_values: List[float]) -> List[float]:
        """Find breakeven points where P&L crosses zero"""
        breakevens = []
        
        for i in range(len(pnl_values) - 1):
            if (pnl_values[i] <= 0 <= pnl_values[i + 1]) or (pnl_values[i] >= 0 >= pnl_values[i + 1]):
                # Linear interpolation to find exact breakeven
                x1, y1 = price_range[i], pnl_values[i]
                x2, y2 = price_range[i + 1], pnl_values[i + 1]
                
                if y2 != y1:  # Avoid division by zero
                    breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                    breakevens.append(breakeven)
        
        return breakevens
    
    def _calculate_probability_of_profit(self, current_price: float, breakeven_prices: List[float],
                                       iv: float, days_to_exp: int) -> float:
        """Calculate probability of profit using Black-Scholes model"""
        if not breakeven_prices:
            return 0.5
        
        # Simplified calculation using log-normal distribution
        # In production, use more sophisticated option pricing models
        
        time_to_exp = days_to_exp / 365.0
        
        if time_to_exp <= 0:
            return 1.0 if current_price > min(breakeven_prices) else 0.0
        
        # Use the lower breakeven for call debit spreads
        target_price = min(breakeven_prices)
        
        # Calculate probability using log-normal distribution
        d = (np.log(target_price / current_price)) / (iv * np.sqrt(time_to_exp))
        
        from scipy.stats import norm
        prob = 1 - norm.cdf(d)
        
        return max(0.0, min(1.0, prob))
    
    def export_optimization_results(self, results: OptimizationResult, filename: str = "optimization_results.json"):
        """Export optimization results to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'best_params': results.best_params,
                'best_value': results.best_value,
                'backtest_results': results.backtest_results,
                'payoff_diagram_data': results.payoff_diagram_data,
                'optimization_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Optimization results exported to {filename}")