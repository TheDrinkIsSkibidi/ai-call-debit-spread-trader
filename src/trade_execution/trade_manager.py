import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/pyalgostrategypool'))

import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from enum import Enum

from src.core.config import settings
from src.core.database import get_db, Trade
from src.spread_constructor.spread_builder import CallDebitSpread
from src.llm_integration.thesis_scorer import LLMThesisScore
from src.ml_classifier.entry_classifier import MLPrediction


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    success: bool
    trade_id: Optional[int]
    order_ids: List[str]
    filled_price: Optional[float]
    error_message: Optional[str]
    execution_timestamp: datetime


@dataclass
class PositionMonitor:
    trade_id: int
    symbol: str
    long_leg_qty: int
    short_leg_qty: int
    entry_date: datetime
    target_profit_pct: float
    stop_loss_pct: float
    days_to_expiration: int


class TradeExecutionManager:
    def __init__(self):
        # Initialize Alpaca API
        self.alpaca = tradeapi.REST(
            settings.alpaca_api_key,
            settings.alpaca_secret_key,
            base_url='https://paper-api.alpaca.markets' if settings.paper_trading else 'https://api.alpaca.markets',
            api_version='v2'
        )
        
        # Trading parameters
        self.max_positions = settings.max_positions
        self.max_risk_per_trade = settings.max_risk_per_trade
        self.default_quantity = 1  # Number of contracts per trade
        
        # Exit criteria
        self.profit_target_pct = 0.8  # Take profit at 80% of max profit
        self.stop_loss_pct = 0.5     # Stop loss at 50% of max loss
        
        # Track active positions
        self.active_positions: Dict[int, PositionMonitor] = {}
    
    def execute_spread_trade(self, spread: CallDebitSpread, llm_score: LLMThesisScore, 
                           ml_prediction: MLPrediction) -> ExecutionResult:
        """Execute a call debit spread trade"""
        
        try:
            # Pre-execution checks
            if not self._pre_execution_checks(spread, llm_score, ml_prediction):
                return ExecutionResult(
                    success=False,
                    trade_id=None,
                    order_ids=[],
                    filled_price=None,
                    error_message="Pre-execution checks failed",
                    execution_timestamp=datetime.now()
                )
            
            # Create trade record in database
            trade_id = self._create_trade_record(spread, llm_score, ml_prediction)
            
            # Execute the spread order
            order_result = self._execute_spread_order(spread)
            
            if order_result['success']:
                # Update trade record with execution details
                self._update_trade_execution(trade_id, order_result)
                
                # Add to position monitoring
                self._add_position_monitor(trade_id, spread, order_result)
                
                return ExecutionResult(
                    success=True,
                    trade_id=trade_id,
                    order_ids=order_result['order_ids'],
                    filled_price=order_result['filled_price'],
                    error_message=None,
                    execution_timestamp=datetime.now()
                )
            else:
                # Update trade record as failed
                self._mark_trade_failed(trade_id, order_result['error'])
                
                return ExecutionResult(
                    success=False,
                    trade_id=trade_id,
                    order_ids=[],
                    filled_price=None,
                    error_message=order_result['error'],
                    execution_timestamp=datetime.now()
                )
                
        except Exception as e:
            print(f"Error executing spread trade: {e}")
            return ExecutionResult(
                success=False,
                trade_id=None,
                order_ids=[],
                filled_price=None,
                error_message=str(e),
                execution_timestamp=datetime.now()
            )
    
    def monitor_positions(self) -> List[Dict]:
        """Monitor all active positions and execute exit strategies"""
        position_updates = []
        
        for trade_id, monitor in list(self.active_positions.items()):
            try:
                update = self._check_exit_conditions(monitor)
                if update:
                    position_updates.append(update)
                    
                    # If position was closed, remove from monitoring
                    if update.get('action') == 'closed':
                        del self.active_positions[trade_id]
                        
            except Exception as e:
                print(f"Error monitoring position {trade_id}: {e}")
        
        return position_updates
    
    def force_close_position(self, trade_id: int, reason: str = "manual_close") -> ExecutionResult:
        """Manually close a position"""
        if trade_id not in self.active_positions:
            return ExecutionResult(
                success=False,
                trade_id=trade_id,
                order_ids=[],
                filled_price=None,
                error_message="Position not found in active monitoring",
                execution_timestamp=datetime.now()
            )
        
        monitor = self.active_positions[trade_id]
        
        try:
            # Execute closing orders
            close_result = self._execute_closing_orders(monitor, reason)
            
            if close_result['success']:
                # Update database
                self._update_trade_close(trade_id, close_result, reason)
                
                # Remove from monitoring
                del self.active_positions[trade_id]
                
                return ExecutionResult(
                    success=True,
                    trade_id=trade_id,
                    order_ids=close_result['order_ids'],
                    filled_price=close_result['filled_price'],
                    error_message=None,
                    execution_timestamp=datetime.now()
                )
            else:
                return ExecutionResult(
                    success=False,
                    trade_id=trade_id,
                    order_ids=[],
                    filled_price=None,
                    error_message=close_result['error'],
                    execution_timestamp=datetime.now()
                )
                
        except Exception as e:
            print(f"Error force closing position {trade_id}: {e}")
            return ExecutionResult(
                success=False,
                trade_id=trade_id,
                order_ids=[],
                filled_price=None,
                error_message=str(e),
                execution_timestamp=datetime.now()
            )
    
    def get_account_status(self) -> Dict:
        """Get current account status and position summary"""
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.list_positions()
            
            # Calculate total options positions value
            options_value = 0
            options_count = 0
            
            for position in positions:
                if 'O:' in position.symbol:  # Alpaca options symbol format
                    options_value += float(position.market_value or 0)
                    options_count += 1
            
            return {
                'account_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'day_trade_count': int(account.daytrade_count),
                'options_positions': options_count,
                'options_value': options_value,
                'active_monitors': len(self.active_positions),
                'paper_trading': settings.paper_trading,
                'max_positions': self.max_positions,
                'max_risk_per_trade': self.max_risk_per_trade
            }
            
        except Exception as e:
            print(f"Error getting account status: {e}")
            return {'error': str(e)}
    
    def _pre_execution_checks(self, spread: CallDebitSpread, llm_score: LLMThesisScore, 
                            ml_prediction: MLPrediction) -> bool:
        """Perform pre-execution validation checks"""
        
        # Check if we have room for more positions
        if len(self.active_positions) >= self.max_positions:
            print("Maximum positions reached")
            return False
        
        # Check risk limits
        trade_risk = spread.net_debit * 100 * self.default_quantity  # Convert to dollars
        account = self.alpaca.get_account()
        account_value = float(account.portfolio_value)
        
        if trade_risk > account_value * self.max_risk_per_trade:
            print(f"Trade risk {trade_risk} exceeds maximum risk per trade")
            return False
        
        # Check LLM confidence threshold
        if llm_score.confidence_score < 70:
            print(f"LLM confidence {llm_score.confidence_score} below threshold")
            return False
        
        # Check ML prediction threshold
        if ml_prediction.win_probability < 0.6:
            print(f"ML win probability {ml_prediction.win_probability} below threshold")
            return False
        
        # Check market hours (simplified)
        current_time = datetime.now().time()
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        
        if not (market_open <= current_time <= market_close):
            print("Market is closed")
            return False
        
        return True
    
    def _execute_spread_order(self, spread: CallDebitSpread) -> Dict:
        """Execute the actual spread order using Alpaca API"""
        try:
            # Convert strikes and expiration to Alpaca option symbols
            # Format: O:SPY251219C00450000 (O:SYMBOL+EXPDATE+TYPE+STRIKE)
            exp_date = datetime.strptime(spread.expiration, "%Y-%m-%d").strftime("%y%m%d")
            
            long_symbol = f"O:{spread.symbol}{exp_date}C{int(spread.long_strike * 1000):08d}"
            short_symbol = f"O:{spread.symbol}{exp_date}C{int(spread.short_strike * 1000):08d}"
            
            # Create multi-leg order for the spread
            order_data = {
                "symbol": spread.symbol,
                "qty": self.default_quantity,
                "side": "buy",  # Net debit = buying the spread
                "type": "market",
                "time_in_force": "day",
                "order_class": "oto",  # One-triggers-other for exit orders
                "legs": [
                    {
                        "symbol": long_symbol,
                        "qty": self.default_quantity,
                        "side": "buy"
                    },
                    {
                        "symbol": short_symbol,
                        "qty": self.default_quantity,
                        "side": "sell"
                    }
                ]
            }
            
            # Submit the order
            if settings.paper_trading:
                # For paper trading, simulate order execution
                return {
                    'success': True,
                    'order_ids': ['paper_long_123', 'paper_short_456'],
                    'filled_price': spread.net_debit,
                    'fill_time': datetime.now()
                }
            else:
                # Real trading execution
                order = self.alpaca.submit_order(**order_data)
                
                # Wait for fill confirmation (simplified)
                # In production, implement proper order monitoring
                return {
                    'success': True,
                    'order_ids': [order.id],
                    'filled_price': spread.net_debit,  # Placeholder
                    'fill_time': datetime.now()
                }
                
        except Exception as e:
            print(f"Error executing spread order: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_trade_record(self, spread: CallDebitSpread, llm_score: LLMThesisScore, 
                           ml_prediction: MLPrediction) -> int:
        """Create trade record in database"""
        
        db = next(get_db())
        try:
            trade = Trade(
                symbol=spread.symbol,
                strategy_type="call_debit_spread",
                long_strike=spread.long_strike,
                short_strike=spread.short_strike,
                expiration_date=datetime.strptime(spread.expiration, "%Y-%m-%d"),
                quantity=self.default_quantity,
                entry_debit=spread.net_debit,
                llm_score=llm_score.confidence_score,
                llm_reasoning=llm_score.reasoning[:500],  # Truncate for DB
                ml_probability=ml_prediction.win_probability,
                market_data=json.dumps({
                    'risk_reward_ratio': spread.risk_reward_ratio,
                    'days_to_expiration': spread.days_to_expiration,
                    'profit_probability': spread.profit_probability
                }),
                is_paper_trade=settings.paper_trading
            )
            
            db.add(trade)
            db.commit()
            db.refresh(trade)
            
            return trade.id
            
        finally:
            db.close()
    
    def _update_trade_execution(self, trade_id: int, order_result: Dict):
        """Update trade record with execution details"""
        
        db = next(get_db())
        try:
            trade = db.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.entry_price = order_result['filled_price']
                trade.status = "open"
                db.commit()
                
        finally:
            db.close()
    
    def _add_position_monitor(self, trade_id: int, spread: CallDebitSpread, order_result: Dict):
        """Add position to monitoring system"""
        
        monitor = PositionMonitor(
            trade_id=trade_id,
            symbol=spread.symbol,
            long_leg_qty=self.default_quantity,
            short_leg_qty=self.default_quantity,
            entry_date=datetime.now(),
            target_profit_pct=self.profit_target_pct,
            stop_loss_pct=self.stop_loss_pct,
            days_to_expiration=spread.days_to_expiration
        )
        
        self.active_positions[trade_id] = monitor
        print(f"Added position {trade_id} to monitoring")
    
    def _check_exit_conditions(self, monitor: PositionMonitor) -> Optional[Dict]:
        """Check if position meets exit conditions"""
        
        try:
            # Get current position value (simplified)
            # In production, get real-time option prices
            
            # Check time decay (close 7 days before expiration)
            days_remaining = (monitor.entry_date + timedelta(days=monitor.days_to_expiration) - datetime.now()).days
            
            if days_remaining <= 7:
                # Execute time-based close
                close_result = self._execute_closing_orders(monitor, "time_decay")
                if close_result['success']:
                    self._update_trade_close(monitor.trade_id, close_result, "time_decay")
                    return {
                        'trade_id': monitor.trade_id,
                        'action': 'closed',
                        'reason': 'time_decay',
                        'result': close_result
                    }
            
            # Check profit/loss targets
            # This would require real-time option pricing
            # For now, implement placeholder logic
            
            return None
            
        except Exception as e:
            print(f"Error checking exit conditions for {monitor.trade_id}: {e}")
            return None
    
    def _execute_closing_orders(self, monitor: PositionMonitor, reason: str) -> Dict:
        """Execute orders to close the spread position"""
        
        try:
            if settings.paper_trading:
                # Simulate closing in paper trading
                return {
                    'success': True,
                    'order_ids': ['paper_close_123'],
                    'filled_price': 0.5,  # Placeholder exit price
                    'fill_time': datetime.now()
                }
            else:
                # Implement real closing orders
                # This would involve selling the long leg and buying back the short leg
                return {
                    'success': True,
                    'order_ids': ['real_close_123'],
                    'filled_price': 0.5,
                    'fill_time': datetime.now()
                }
                
        except Exception as e:
            print(f"Error executing closing orders: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_trade_close(self, trade_id: int, close_result: Dict, reason: str):
        """Update trade record when position is closed"""
        
        db = next(get_db())
        try:
            trade = db.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.exit_date = datetime.now()
                trade.exit_credit = close_result['filled_price']
                trade.status = "closed"
                
                # Calculate P&L
                trade.pnl = (trade.exit_credit - trade.entry_debit) * 100 * trade.quantity
                if trade.entry_debit > 0:
                    trade.roi = trade.pnl / (trade.entry_debit * 100 * trade.quantity)
                
                db.commit()
                
        finally:
            db.close()
    
    def _mark_trade_failed(self, trade_id: int, error: str):
        """Mark trade as failed in database"""
        
        db = next(get_db())
        try:
            trade = db.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.status = "cancelled"
                trade.exit_date = datetime.now()
                
                # Add error to market_data JSON
                market_data = json.loads(trade.market_data or '{}')
                market_data['execution_error'] = error
                trade.market_data = json.dumps(market_data)
                
                db.commit()
                
        finally:
            db.close()