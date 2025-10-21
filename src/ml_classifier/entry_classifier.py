import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from src.spread_constructor.spread_builder import CallDebitSpread
from src.data_ingestion.market_data import MarketSnapshot


@dataclass
class MLPrediction:
    win_probability: float
    confidence: float
    feature_importance: Dict[str, float]
    model_version: str


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    cross_val_scores: List[float]


class MLEntryClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_version = "1.0.0"
        self.trained_date = None
        
        # Define feature columns
        self.feature_columns = [
            'current_price', 'rsi', 'ema_8', 'ema_21', 'iv_rank', 'volume',
            'long_strike', 'short_strike', 'net_debit', 'max_profit', 'max_loss',
            'breakeven', 'risk_reward_ratio', 'days_to_expiration',
            'long_delta', 'short_delta', 'net_delta', 'long_iv', 'short_iv',
            'strike_width', 'debit_percentage', 'price_to_long_strike_ratio',
            'price_to_short_strike_ratio', 'price_to_breakeven_ratio',
            'ema_trend', 'rsi_regime', 'iv_rank_regime'
        ]
    
    def prepare_features(self, spread: CallDebitSpread, market_data: MarketSnapshot) -> np.ndarray:
        """Prepare feature vector for a single spread"""
        
        # Basic market features
        features = {
            'current_price': market_data.price,
            'rsi': market_data.rsi or 50.0,
            'ema_8': market_data.ema_8 or market_data.price,
            'ema_21': market_data.ema_21 or market_data.price,
            'iv_rank': market_data.iv_rank or 50.0,
            'volume': market_data.volume or 0,
        }
        
        # Spread features
        features.update({
            'long_strike': spread.long_strike,
            'short_strike': spread.short_strike,
            'net_debit': spread.net_debit,
            'max_profit': spread.max_profit,
            'max_loss': spread.max_loss,
            'breakeven': spread.breakeven,
            'risk_reward_ratio': spread.risk_reward_ratio,
            'days_to_expiration': spread.days_to_expiration,
            'long_delta': spread.long_option.delta,
            'short_delta': spread.short_option.delta,
            'net_delta': spread.long_option.delta - spread.short_option.delta,
            'long_iv': spread.long_option.implied_volatility,
            'short_iv': spread.short_option.implied_volatility,
        })
        
        # Derived features
        strike_width = spread.short_strike - spread.long_strike
        features.update({
            'strike_width': strike_width,
            'debit_percentage': spread.net_debit / strike_width if strike_width > 0 else 0,
            'price_to_long_strike_ratio': market_data.price / spread.long_strike if spread.long_strike > 0 else 0,
            'price_to_short_strike_ratio': market_data.price / spread.short_strike if spread.short_strike > 0 else 0,
            'price_to_breakeven_ratio': market_data.price / spread.breakeven if spread.breakeven > 0 else 0,
        })
        
        # Technical indicator regimes
        ema_8 = features['ema_8']
        ema_21 = features['ema_21']
        features.update({
            'ema_trend': 1 if ema_8 > ema_21 else 0,
            'rsi_regime': 1 if features['rsi'] > 70 else (0 if features['rsi'] < 30 else 0.5),
            'iv_rank_regime': 1 if features['iv_rank'] > 70 else (0 if features['iv_rank'] < 30 else 0.5),
        })
        
        # Ensure all features are present and handle missing values
        feature_vector = []
        for col in self.feature_columns:
            value = features.get(col, 0.0)
            if pd.isna(value) or value is None:
                value = 0.0
            feature_vector.append(float(value))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def train_model(self, trade_history: List[Dict], retrain: bool = False) -> ModelMetrics:
        """Train the ML model using historical trade data"""
        
        if not trade_history:
            raise ValueError("No trade history provided for training")
        
        # Convert trade history to DataFrame
        df = pd.DataFrame(trade_history)
        
        # Prepare features and labels
        X, y = self._prepare_training_data(df)
        
        if len(X) < 50:
            raise ValueError("Insufficient training data. Need at least 50 historical trades.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with scaling and gradient boosting
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                validation_fraction=0.2,
                n_iter_no_change=10
            ))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        self.feature_names = self.feature_columns
        self.trained_date = datetime.now()
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1_score=f1_score(y_test, y_pred),
            auc_roc=roc_auc_score(y_test, y_pred_proba),
            cross_val_scores=cross_val_score(self.model, X_train, y_train, cv=5).tolist()
        )
        
        print(f"Model Training Complete:")
        print(f"Accuracy: {metrics.accuracy:.3f}")
        print(f"Precision: {metrics.precision:.3f}")
        print(f"Recall: {metrics.recall:.3f}")
        print(f"F1-Score: {metrics.f1_score:.3f}")
        print(f"AUC-ROC: {metrics.auc_roc:.3f}")
        print(f"Cross-validation scores: {[f'{score:.3f}' for score in metrics.cross_val_scores]}")
        
        return metrics
    
    def predict_entry(self, spread: CallDebitSpread, market_data: MarketSnapshot) -> MLPrediction:
        """Predict probability of successful trade entry"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare features
        X = self.prepare_features(spread, market_data)
        
        # Make prediction
        win_probability = self.model.predict_proba(X)[0, 1]
        
        # Calculate confidence based on prediction certainty
        confidence = abs(win_probability - 0.5) * 2  # 0.5 = maximum uncertainty
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
        
        return MLPrediction(
            win_probability=win_probability,
            confidence=confidence,
            feature_importance=feature_importance,
            model_version=self.model_version
        )
    
    def batch_predict(self, spreads_with_market_data: List[Tuple[CallDebitSpread, MarketSnapshot]]) -> List[MLPrediction]:
        """Predict for multiple spreads"""
        predictions = []
        
        for spread, market_data in spreads_with_market_data:
            try:
                prediction = self.predict_entry(spread, market_data)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting for {spread.symbol}: {e}")
                # Add default prediction
                predictions.append(MLPrediction(
                    win_probability=0.5,
                    confidence=0.0,
                    feature_importance={},
                    model_version=self.model_version
                ))
        
        return predictions
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from trade history"""
        
        # Create target variable (1 if ROI > 50%, 0 otherwise)
        y = (df['roi'] > 0.5).astype(int)
        
        # Prepare features
        feature_data = []
        
        for _, row in df.iterrows():
            # Reconstruct spread and market data from trade record
            spread_dict = {
                'symbol': row.get('symbol', ''),
                'expiration': row.get('expiration_date', ''),
                'long_strike': row.get('long_strike', 0),
                'short_strike': row.get('short_strike', 0),
                'net_debit': row.get('entry_debit', 0),
                'max_profit': row.get('long_strike', 0) - row.get('short_strike', 0) - row.get('entry_debit', 0),
                'max_loss': row.get('entry_debit', 0),
                'breakeven': row.get('long_strike', 0) + row.get('entry_debit', 0),
                'risk_reward_ratio': 1.0,  # Placeholder
                'days_to_expiration': 30,  # Placeholder
                'long_option': {'delta': 0.5, 'implied_volatility': 0.3},
                'short_option': {'delta': 0.3, 'implied_volatility': 0.25}
            }
            
            market_dict = {
                'price': row.get('entry_price', 0),
                'rsi': 50.0,  # Placeholder - use market_data JSON if available
                'ema_8': row.get('entry_price', 0),
                'ema_21': row.get('entry_price', 0),
                'iv_rank': 50.0,
                'volume': 1000000
            }
            
            # Extract from market_data JSON if available
            if 'market_data' in row and row['market_data']:
                try:
                    market_json = json.loads(row['market_data']) if isinstance(row['market_data'], str) else row['market_data']
                    market_dict.update(market_json)
                except:
                    pass
            
            # Create mock objects for feature extraction
            from types import SimpleNamespace
            
            spread = SimpleNamespace(**spread_dict)
            spread.long_option = SimpleNamespace(**spread_dict['long_option'])
            spread.short_option = SimpleNamespace(**spread_dict['short_option'])
            
            market_data = SimpleNamespace(**market_dict)
            
            # Extract features
            features = self.prepare_features(spread, market_data)
            feature_data.append(features.flatten())
        
        X = np.array(feature_data)
        
        return X, y.values
    
    def save_model(self, filepath: str = "ml_entry_classifier.joblib"):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'trained_date': self.trained_date,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "ml_entry_classifier.joblib"):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_version = model_data['model_version']
            self.trained_date = model_data['trained_date']
            self.feature_columns = model_data.get('feature_columns', self.feature_columns)
            
            print(f"Model loaded from {filepath}")
            print(f"Model version: {self.model_version}")
            print(f"Trained on: {self.trained_date}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get top N most important features"""
        if self.model is None:
            return {}
        
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            
            # Sort and get top N
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features[:top_n])
        
        return {}
    
    def model_diagnostics(self) -> Dict:
        """Get model diagnostics and statistics"""
        if self.model is None:
            return {"error": "No model trained"}
        
        diagnostics = {
            'model_version': self.model_version,
            'trained_date': self.trained_date.isoformat() if self.trained_date else None,
            'feature_count': len(self.feature_names),
            'top_features': self.get_feature_importance(5),
            'model_params': self.model.named_steps['classifier'].get_params()
        }
        
        return diagnostics