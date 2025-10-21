import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from src.core.database import get_db, Trade, AIInsight
from src.llm_integration.thesis_scorer import LLMThesisScorer


@dataclass
class PerformanceMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    total_pnl: float
    total_roi: float
    max_win: float
    max_loss: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


@dataclass
class TradeAnalysis:
    performance_metrics: PerformanceMetrics
    monthly_performance: pd.DataFrame
    trade_distribution: Dict
    correlation_analysis: Dict
    ai_model_performance: Dict


class TradeJournalAnalyzer:
    def __init__(self):
        self.llm_scorer = LLMThesisScorer()
    
    def get_all_trades(self, days_back: Optional[int] = None) -> pd.DataFrame:
        """Retrieve all trades from database as DataFrame"""
        
        db = next(get_db())
        try:
            query = db.query(Trade)
            
            if days_back:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                query = query.filter(Trade.entry_date >= cutoff_date)
            
            trades = query.all()
            
            # Convert to DataFrame
            trade_data = []
            for trade in trades:
                trade_dict = {
                    'id': trade.id,
                    'symbol': trade.symbol,
                    'strategy_type': trade.strategy_type,
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'long_strike': trade.long_strike,
                    'short_strike': trade.short_strike,
                    'expiration_date': trade.expiration_date,
                    'quantity': trade.quantity,
                    'entry_debit': trade.entry_debit,
                    'exit_credit': trade.exit_credit,
                    'pnl': trade.pnl,
                    'roi': trade.roi,
                    'llm_score': trade.llm_score,
                    'ml_probability': trade.ml_probability,
                    'status': trade.status,
                    'is_paper_trade': trade.is_paper_trade,
                    'market_data': trade.market_data
                }
                
                # Parse market_data JSON
                if trade.market_data:
                    try:
                        market_json = json.loads(trade.market_data)
                        trade_dict.update(market_json)
                    except:
                        pass
                
                trade_data.append(trade_dict)
            
            return pd.DataFrame(trade_data)
            
        finally:
            db.close()
    
    def calculate_performance_metrics(self, trades_df: pd.DataFrame) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if trades_df.empty:
            return self._empty_metrics()
        
        # Filter only closed trades for analysis
        closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return self._empty_metrics()
        
        # Basic counts
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        wins = closed_trades[closed_trades['pnl'] > 0]['pnl']
        losses = closed_trades[closed_trades['pnl'] <= 0]['pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # ROI metrics
        win_rois = closed_trades[closed_trades['roi'] > 0]['roi']
        loss_rois = closed_trades[closed_trades['roi'] <= 0]['roi']
        
        avg_win_pct = win_rois.mean() if len(win_rois) > 0 else 0
        avg_loss_pct = loss_rois.mean() if len(loss_rois) > 0 else 0
        
        # Total metrics
        total_pnl = closed_trades['pnl'].sum()
        total_roi = closed_trades['roi'].mean()
        
        # Extremes
        max_win = closed_trades['pnl'].max()
        max_loss = closed_trades['pnl'].min()
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Drawdown analysis
        cumulative_pnl = closed_trades['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()
        
        # Risk-adjusted returns
        returns = closed_trades['roi'].fillna(0)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
        sortino_ratio = returns.mean() / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_roi / (max_drawdown / total_pnl) if max_drawdown > 0 and total_pnl != 0 else 0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            total_pnl=total_pnl,
            total_roi=total_roi,
            max_win=max_win,
            max_loss=max_loss,
            expectancy=expectancy,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio
        )
    
    def analyze_monthly_performance(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze performance by month"""
        
        if trades_df.empty:
            return pd.DataFrame()
        
        closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return pd.DataFrame()
        
        # Extract month from exit_date
        closed_trades['month'] = pd.to_datetime(closed_trades['exit_date']).dt.to_period('M')
        
        # Group by month and calculate metrics
        monthly_stats = closed_trades.groupby('month').agg({
            'pnl': ['count', 'sum', 'mean'],
            'roi': 'mean'
        }).round(4)
        
        # Flatten column names
        monthly_stats.columns = ['trades_count', 'total_pnl', 'avg_pnl', 'avg_roi']
        
        # Calculate win rate by month
        win_rates = closed_trades.groupby('month').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x)
        ).rename('win_rate')
        
        monthly_stats = monthly_stats.join(win_rates)
        
        return monthly_stats.reset_index()
    
    def analyze_trade_distribution(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze trade distribution across various dimensions"""
        
        if trades_df.empty:
            return {}
        
        closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
        
        analysis = {}
        
        # By symbol
        symbol_stats = closed_trades.groupby('symbol').agg({
            'pnl': ['count', 'sum', 'mean'],
            'roi': 'mean'
        }).round(4)
        analysis['by_symbol'] = symbol_stats.to_dict()
        
        # By LLM score ranges
        if 'llm_score' in closed_trades.columns:
            closed_trades['llm_score_range'] = pd.cut(
                closed_trades['llm_score'], 
                bins=[0, 70, 80, 90, 100], 
                labels=['<70', '70-80', '80-90', '90+']
            )
            
            llm_stats = closed_trades.groupby('llm_score_range').agg({
                'pnl': ['count', 'sum', 'mean'],
                'roi': 'mean'
            }).round(4)
            analysis['by_llm_score'] = llm_stats.to_dict()
        
        # By ML prediction ranges
        if 'ml_probability' in closed_trades.columns:
            closed_trades['ml_prob_range'] = pd.cut(
                closed_trades['ml_probability'], 
                bins=[0, 0.6, 0.7, 0.8, 1.0], 
                labels=['<60%', '60-70%', '70-80%', '80%+']
            )
            
            ml_stats = closed_trades.groupby('ml_prob_range').agg({
                'pnl': ['count', 'sum', 'mean'],
                'roi': 'mean'
            }).round(4)
            analysis['by_ml_prediction'] = ml_stats.to_dict()
        
        # By days to expiration
        if 'days_to_expiration' in closed_trades.columns:
            closed_trades['dte_range'] = pd.cut(
                closed_trades['days_to_expiration'], 
                bins=[0, 21, 35, 45, 100], 
                labels=['<21', '21-35', '35-45', '45+']
            )
            
            dte_stats = closed_trades.groupby('dte_range').agg({
                'pnl': ['count', 'sum', 'mean'],
                'roi': 'mean'
            }).round(4)
            analysis['by_dte'] = dte_stats.to_dict()
        
        return analysis
    
    def analyze_ai_model_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance of AI models (LLM and ML)"""
        
        if trades_df.empty:
            return {}
        
        closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return {}
        
        analysis = {}
        
        # LLM Score Analysis
        if 'llm_score' in closed_trades.columns:
            # Correlation between LLM score and actual outcomes
            llm_correlation = closed_trades['llm_score'].corr(closed_trades['roi'])
            
            # Accuracy by score ranges
            llm_accuracy = {}
            for threshold in [70, 75, 80, 85, 90]:
                high_score_trades = closed_trades[closed_trades['llm_score'] >= threshold]
                if len(high_score_trades) > 0:
                    win_rate = (high_score_trades['pnl'] > 0).mean()
                    llm_accuracy[f'score_{threshold}+'] = {
                        'trades': len(high_score_trades),
                        'win_rate': win_rate,
                        'avg_roi': high_score_trades['roi'].mean()
                    }
            
            analysis['llm_performance'] = {
                'correlation_with_roi': llm_correlation,
                'accuracy_by_threshold': llm_accuracy
            }
        
        # ML Model Analysis
        if 'ml_probability' in closed_trades.columns:
            # Correlation between ML prediction and actual outcomes
            ml_correlation = closed_trades['ml_probability'].corr(closed_trades['roi'])
            
            # Calibration analysis
            ml_calibration = {}
            for threshold in [0.6, 0.65, 0.7, 0.75, 0.8]:
                high_prob_trades = closed_trades[closed_trades['ml_probability'] >= threshold]
                if len(high_prob_trades) > 0:
                    actual_win_rate = (high_prob_trades['pnl'] > 0).mean()
                    predicted_prob = high_prob_trades['ml_probability'].mean()
                    calibration_error = abs(actual_win_rate - predicted_prob)
                    
                    ml_calibration[f'prob_{threshold}+'] = {
                        'trades': len(high_prob_trades),
                        'predicted_prob': predicted_prob,
                        'actual_win_rate': actual_win_rate,
                        'calibration_error': calibration_error,
                        'avg_roi': high_prob_trades['roi'].mean()
                    }
            
            analysis['ml_performance'] = {
                'correlation_with_roi': ml_correlation,
                'calibration_analysis': ml_calibration
            }
        
        # Combined AI Signal Analysis
        if 'llm_score' in closed_trades.columns and 'ml_probability' in closed_trades.columns:
            # High confidence trades (both AI signals strong)
            high_confidence = closed_trades[
                (closed_trades['llm_score'] >= 80) & 
                (closed_trades['ml_probability'] >= 0.7)
            ]
            
            # Low confidence trades
            low_confidence = closed_trades[
                (closed_trades['llm_score'] < 75) | 
                (closed_trades['ml_probability'] < 0.65)
            ]
            
            analysis['combined_ai_signals'] = {
                'high_confidence': {
                    'trades': len(high_confidence),
                    'win_rate': (high_confidence['pnl'] > 0).mean() if len(high_confidence) > 0 else 0,
                    'avg_roi': high_confidence['roi'].mean() if len(high_confidence) > 0 else 0
                },
                'low_confidence': {
                    'trades': len(low_confidence),
                    'win_rate': (low_confidence['pnl'] > 0).mean() if len(low_confidence) > 0 else 0,
                    'avg_roi': low_confidence['roi'].mean() if len(low_confidence) > 0 else 0
                }
            }
        
        return analysis
    
    def generate_equity_curve(self, trades_df: pd.DataFrame, save_path: str = "equity_curve.png") -> str:
        """Generate and save equity curve visualization"""
        
        closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return "No closed trades to plot"
        
        # Sort by exit date
        closed_trades = closed_trades.sort_values('exit_date')
        
        # Calculate cumulative P&L
        closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Main equity curve
        plt.subplot(2, 1, 1)
        plt.plot(closed_trades['exit_date'], closed_trades['cumulative_pnl'], 
                linewidth=2, color='blue', label='Cumulative P&L')
        plt.title('Equity Curve', fontsize=16, fontweight='bold')
        plt.ylabel('Cumulative P&L ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Monthly returns
        plt.subplot(2, 1, 2)
        monthly_returns = self.analyze_monthly_performance(trades_df)
        if not monthly_returns.empty:
            plt.bar(range(len(monthly_returns)), monthly_returns['total_pnl'], 
                   color=['green' if x > 0 else 'red' for x in monthly_returns['total_pnl']])
            plt.title('Monthly P&L', fontsize=14)
            plt.ylabel('Monthly P&L ($)', fontsize=12)
            plt.xticks(range(len(monthly_returns)), 
                      [str(m) for m in monthly_returns['month']], rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Equity curve saved to {save_path}"
    
    def generate_ai_insights(self, trades_df: pd.DataFrame, save_to_db: bool = True) -> Dict:
        """Generate AI-powered insights about trading patterns"""
        
        if trades_df.empty:
            return {"error": "No trade data available"}
        
        # Prepare trade history for LLM analysis
        trade_history = trades_df.to_dict('records')
        
        # Use LLM to analyze patterns
        insights = self.llm_scorer.analyze_portfolio_patterns(trade_history[-50:])  # Last 50 trades
        
        # Add quantitative analysis
        performance_metrics = self.calculate_performance_metrics(trades_df)
        ai_performance = self.analyze_ai_model_performance(trades_df)
        
        comprehensive_insights = {
            'llm_insights': insights,
            'performance_summary': performance_metrics.__dict__,
            'ai_model_analysis': ai_performance,
            'key_statistics': {
                'total_trades': len(trades_df),
                'closed_trades': len(trades_df[trades_df['status'] == 'closed']),
                'current_win_rate': performance_metrics.win_rate,
                'expectancy': performance_metrics.expectancy,
                'profit_factor': performance_metrics.profit_factor
            },
            'generated_date': datetime.now().isoformat()
        }
        
        # Save to database if requested
        if save_to_db:
            self._save_insights_to_db(comprehensive_insights, trades_df)
        
        return comprehensive_insights
    
    def _save_insights_to_db(self, insights: Dict, trades_df: pd.DataFrame):
        """Save AI insights to database"""
        
        db = next(get_db())
        try:
            # Calculate date range
            if not trades_df.empty:
                start_date = trades_df['entry_date'].min()
                end_date = trades_df['entry_date'].max()
            else:
                start_date = end_date = datetime.now()
            
            # Extract key metrics
            perf_summary = insights.get('performance_summary', {})
            
            ai_insight = AIInsight(
                start_date=start_date,
                end_date=end_date,
                patterns_identified=insights.get('llm_insights', {}),
                recommendations=insights.get('ai_model_analysis', {}),
                confidence_adjustments=insights.get('key_statistics', {}),
                total_trades=perf_summary.get('total_trades', 0),
                win_rate=perf_summary.get('win_rate', 0),
                avg_roi=perf_summary.get('total_roi', 0)
            )
            
            db.add(ai_insight)
            db.commit()
            
        finally:
            db.close()
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            total_pnl=0.0,
            total_roi=0.0,
            max_win=0.0,
            max_loss=0.0,
            expectancy=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0
        )
    
    def export_trade_log(self, trades_df: pd.DataFrame, filename: str = "spread_log.csv"):
        """Export trades to CSV file"""
        
        if trades_df.empty:
            print("No trades to export")
            return
        
        # Select key columns for export
        export_columns = [
            'id', 'symbol', 'entry_date', 'exit_date', 'long_strike', 'short_strike',
            'expiration_date', 'quantity', 'entry_debit', 'exit_credit', 'pnl', 'roi',
            'llm_score', 'ml_probability', 'status', 'is_paper_trade'
        ]
        
        export_df = trades_df[export_columns].copy()
        export_df.to_csv(filename, index=False)
        
        print(f"Trade log exported to {filename}")
        print(f"Exported {len(export_df)} trades")