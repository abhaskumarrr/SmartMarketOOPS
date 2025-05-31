#!/usr/bin/env python3
"""
Advanced Intelligence System Integration for Enhanced SmartMarketOOPS
Integrates RL agents, meta-learning, automated feature engineering, and sentiment analysis
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .reinforcement_learning_agent import RLTradingSystem
from .meta_learning_ensemble import MetaLearningEnsembleSystem
from .automated_feature_engineering import AutomatedFeatureEngineer
from .sentiment_analysis_system import MarketSentimentAggregator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedIntelligenceSystem:
    """Complete advanced intelligence system integrating all ML components"""
    
    def __init__(self, symbols: List[str] = None):
        """Initialize advanced intelligence system"""
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        
        # Initialize components
        self.rl_system = RLTradingSystem(self.symbols)
        self.meta_learning_system = MetaLearningEnsembleSystem(self.symbols)
        self.feature_engineer = AutomatedFeatureEngineer(max_features=100)
        self.sentiment_aggregator = MarketSentimentAggregator()
        
        # System state
        self.is_initialized = False
        self.performance_history = {}
        self.intelligence_metrics = {}
        
        logger.info(f"Advanced Intelligence System initialized for {len(self.symbols)} symbols")
    
    async def initialize_system(self, historical_data: Dict[str, pd.DataFrame]):
        """Initialize all intelligence components with historical data"""
        logger.info("🧠 Initializing Advanced Intelligence System...")
        
        try:
            # 1. Initialize feature engineering
            logger.info("🔧 Training automated feature engineering...")
            for symbol, data in historical_data.items():
                if len(data) > 100:
                    # Create target for feature engineering (future price movement)
                    target = (data['close'].shift(-1) > data['close']).astype(int)
                    target = target.dropna()
                    
                    # Fit feature engineer
                    engineered_data = self.feature_engineer.fit_transform(data[:-1], target)
                    logger.info(f"✅ Feature engineering trained for {symbol}: {len(engineered_data.columns)} features")
            
            # 2. Initialize meta-learning system
            logger.info("🎯 Training meta-learning ensemble system...")
            self.meta_learning_system.fit_regime_detector(historical_data)
            logger.info("✅ Meta-learning system initialized")
            
            # 3. Initialize RL agents
            logger.info("🤖 Training reinforcement learning agents...")
            for symbol, data in historical_data.items():
                if len(data) > 500:  # Need sufficient data for RL training
                    # Add engineered features to training data
                    enhanced_data = self.feature_engineer.transform(data)
                    
                    # Create and train RL agent
                    agent = self.rl_system.create_agent(symbol, enhanced_data)
                    training_history = self.rl_system.train_agent(symbol, episodes=200)
                    
                    logger.info(f"✅ RL agent trained for {symbol}: "
                               f"Final reward = {np.mean(training_history['episode_rewards'][-10:]):.2f}")
            
            # 4. Initialize performance tracking
            for symbol in self.symbols:
                self.performance_history[symbol] = {
                    'predictions': [],
                    'outcomes': [],
                    'intelligence_scores': []
                }
            
            self.is_initialized = True
            logger.info("🎉 Advanced Intelligence System initialization complete!")
            
        except Exception as e:
            logger.error(f"❌ Error initializing intelligence system: {e}")
            raise
    
    async def get_enhanced_prediction(self, symbol: str, market_data: pd.DataFrame,
                                    base_predictions: Dict[str, float],
                                    base_confidences: Dict[str, float]) -> Dict[str, Any]:
        """Get enhanced prediction using all intelligence components"""
        
        if not self.is_initialized:
            logger.warning("Intelligence system not initialized, returning base predictions")
            return {
                'prediction': np.mean(list(base_predictions.values())),
                'confidence': np.mean(list(base_confidences.values())),
                'enhanced': False
            }
        
        try:
            # 1. Engineer features
            engineered_data = self.feature_engineer.transform(market_data)
            
            # 2. Get RL agent prediction
            rl_prediction = {'action': 'HOLD', 'confidence': 0.0, 'rl_signal': False}
            if symbol in self.rl_system.agents:
                # Create state vector for RL agent
                if len(engineered_data) > 0:
                    # Use latest engineered features as state
                    numeric_features = engineered_data.select_dtypes(include=[np.number]).iloc[-1]
                    state_vector = numeric_features.fillna(0).values[:20]  # Use first 20 features
                    
                    # Pad or truncate to expected size
                    if len(state_vector) < 20:
                        state_vector = np.pad(state_vector, (0, 20 - len(state_vector)))
                    else:
                        state_vector = state_vector[:20]
                    
                    rl_prediction = self.rl_system.get_trading_action(symbol, state_vector)
            
            # 3. Get sentiment analysis
            sentiment_data = await self.sentiment_aggregator.get_comprehensive_sentiment(symbol)
            
            # 4. Combine all predictions for meta-learning
            all_predictions = base_predictions.copy()
            all_confidences = base_confidences.copy()
            
            # Add RL prediction
            if rl_prediction['rl_signal']:
                rl_score = 0.5  # Neutral baseline
                if rl_prediction['action'] == 'BUY':
                    rl_score = 0.5 + (rl_prediction['size'] * 0.5)
                elif rl_prediction['action'] == 'SELL':
                    rl_score = 0.5 - (rl_prediction['size'] * 0.5)
                
                all_predictions['reinforcement_learning'] = rl_score
                all_confidences['reinforcement_learning'] = rl_prediction['confidence']
            
            # Add sentiment prediction
            sentiment_signal = sentiment_data['sentiment_signal']
            sentiment_score = 0.5  # Neutral baseline
            if sentiment_signal['signal'] in ['BUY', 'STRONG_BUY']:
                sentiment_score = 0.5 + (sentiment_signal['strength'] * 0.5)
            elif sentiment_signal['signal'] in ['SELL', 'STRONG_SELL']:
                sentiment_score = 0.5 - (sentiment_signal['strength'] * 0.5)
            
            all_predictions['sentiment_analysis'] = sentiment_score
            all_confidences['sentiment_analysis'] = sentiment_signal['confidence_score']
            
            # 5. Get meta-learning ensemble prediction
            meta_result = self.meta_learning_system.get_enhanced_prediction(
                symbol, market_data, all_predictions, all_confidences
            )
            
            # 6. Calculate intelligence enhancement metrics
            intelligence_metrics = self.calculate_intelligence_metrics(
                base_predictions, all_predictions, meta_result, sentiment_data, rl_prediction
            )
            
            # 7. Create enhanced prediction result
            enhanced_result = {
                'prediction': meta_result['ensemble_prediction'],
                'confidence': meta_result['ensemble_confidence'],
                'quality_score': meta_result['quality_score'],
                'enhanced': True,
                'intelligence_enhanced': True,
                
                # Component predictions
                'base_predictions': base_predictions,
                'rl_prediction': rl_prediction,
                'sentiment_prediction': {
                    'score': sentiment_score,
                    'signal': sentiment_signal,
                    'data': sentiment_data
                },
                'meta_learning_result': meta_result,
                
                # Intelligence metrics
                'intelligence_metrics': intelligence_metrics,
                
                # Feature engineering info
                'engineered_features_count': len(engineered_data.columns),
                'top_features': self.feature_engineer.get_top_features(5),
                
                # System info
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction for {symbol}: {e}")
            # Return base prediction on error
            return {
                'prediction': np.mean(list(base_predictions.values())),
                'confidence': np.mean(list(base_confidences.values())),
                'enhanced': False,
                'error': str(e)
            }
    
    def calculate_intelligence_metrics(self, base_predictions: Dict[str, float],
                                     enhanced_predictions: Dict[str, float],
                                     meta_result: Dict[str, Any],
                                     sentiment_data: Dict[str, Any],
                                     rl_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate intelligence enhancement metrics"""
        
        # Prediction diversity
        all_preds = list(enhanced_predictions.values())
        prediction_diversity = np.std(all_preds) if len(all_preds) > 1 else 0.0
        
        # Confidence improvement
        base_confidence = np.mean(list(base_predictions.values()))
        enhanced_confidence = meta_result['ensemble_confidence']
        confidence_improvement = enhanced_confidence - base_confidence
        
        # Sentiment strength
        sentiment_strength = abs(sentiment_data['aggregate_sentiment'])
        
        # RL contribution
        rl_contribution = rl_prediction['confidence'] if rl_prediction['rl_signal'] else 0.0
        
        # Meta-learning uncertainty
        meta_uncertainty = meta_result.get('uncertainty', 0.0)
        
        # Overall intelligence score
        intelligence_score = (
            (enhanced_confidence * 0.3) +
            (prediction_diversity * 0.2) +
            (sentiment_strength * 0.2) +
            (rl_contribution * 0.2) +
            ((1.0 - meta_uncertainty) * 0.1)
        )
        
        return {
            'intelligence_score': intelligence_score,
            'prediction_diversity': prediction_diversity,
            'confidence_improvement': confidence_improvement,
            'sentiment_strength': sentiment_strength,
            'rl_contribution': rl_contribution,
            'meta_uncertainty': meta_uncertainty,
            'enhancement_factor': enhanced_confidence / max(base_confidence, 0.1)
        }
    
    async def update_performance(self, symbol: str, prediction_result: Dict[str, Any], 
                               actual_outcome: float):
        """Update performance tracking for intelligence system"""
        
        if symbol not in self.performance_history:
            return
        
        try:
            # Store prediction and outcome
            self.performance_history[symbol]['predictions'].append(prediction_result)
            self.performance_history[symbol]['outcomes'].append(actual_outcome)
            
            # Update meta-learning system
            if 'base_predictions' in prediction_result:
                ensemble_pred = prediction_result.get('prediction', 0.5)
                self.meta_learning_system.update_performance(
                    symbol, prediction_result['base_predictions'], ensemble_pred, actual_outcome
                )
            
            # Calculate recent performance
            recent_predictions = self.performance_history[symbol]['predictions'][-50:]
            recent_outcomes = self.performance_history[symbol]['outcomes'][-50:]
            
            if len(recent_predictions) >= 10:
                # Calculate accuracy
                pred_classes = [1 if p.get('prediction', 0.5) > 0.5 else 0 for p in recent_predictions]
                actual_classes = [1 if o > 0.5 else 0 for o in recent_outcomes]
                accuracy = np.mean([p == a for p, a in zip(pred_classes, actual_classes)])
                
                # Calculate intelligence metrics
                intelligence_scores = [p.get('intelligence_metrics', {}).get('intelligence_score', 0.0) 
                                     for p in recent_predictions if 'intelligence_metrics' in p]
                avg_intelligence_score = np.mean(intelligence_scores) if intelligence_scores else 0.0
                
                # Update intelligence metrics
                self.intelligence_metrics[symbol] = {
                    'accuracy': accuracy,
                    'avg_intelligence_score': avg_intelligence_score,
                    'prediction_count': len(recent_predictions),
                    'last_updated': datetime.now().isoformat()
                }
                
                logger.info(f"📊 Intelligence performance for {symbol}: "
                           f"Accuracy={accuracy:.1%}, Intelligence Score={avg_intelligence_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating performance for {symbol}: {e}")
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive intelligence system summary"""
        
        summary = {
            'system_status': 'initialized' if self.is_initialized else 'not_initialized',
            'symbols': self.symbols,
            'components': {
                'reinforcement_learning': {
                    'agents_trained': len(self.rl_system.agents),
                    'training_history': len(self.rl_system.training_history)
                },
                'meta_learning': {
                    'ensembles_active': len(self.meta_learning_system.adaptive_ensembles),
                    'regime_detector_fitted': self.meta_learning_system.regime_detector.is_fitted
                },
                'feature_engineering': {
                    'features_selected': len(self.feature_engineer.selected_features),
                    'is_fitted': self.feature_engineer.is_fitted
                },
                'sentiment_analysis': {
                    'news_sources': len(self.sentiment_aggregator.news_analyzer.news_sources),
                    'sentiment_history': len(self.sentiment_aggregator.aggregated_sentiment_history)
                }
            },
            'performance_metrics': self.intelligence_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    async def save_intelligence_system(self, directory: str):
        """Save complete intelligence system"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        try:
            # Save RL system
            rl_dir = os.path.join(directory, 'reinforcement_learning')
            self.rl_system.save_agents(rl_dir)
            
            # Save meta-learning system
            meta_dir = os.path.join(directory, 'meta_learning')
            self.meta_learning_system.save_system(meta_dir)
            
            # Save feature engineer
            import pickle
            feature_path = os.path.join(directory, 'feature_engineer.pkl')
            with open(feature_path, 'wb') as f:
                pickle.dump(self.feature_engineer, f)
            
            # Save performance history
            performance_path = os.path.join(directory, 'performance_history.pkl')
            with open(performance_path, 'wb') as f:
                pickle.dump(self.performance_history, f)
            
            logger.info(f"💾 Intelligence system saved to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving intelligence system: {e}")
    
    async def load_intelligence_system(self, directory: str):
        """Load complete intelligence system"""
        import os
        
        try:
            # Load RL system
            rl_dir = os.path.join(directory, 'reinforcement_learning')
            if os.path.exists(rl_dir):
                self.rl_system.load_agents(rl_dir)
            
            # Load meta-learning system
            meta_dir = os.path.join(directory, 'meta_learning')
            if os.path.exists(meta_dir):
                self.meta_learning_system.load_system(meta_dir)
            
            # Load feature engineer
            import pickle
            feature_path = os.path.join(directory, 'feature_engineer.pkl')
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    self.feature_engineer = pickle.load(f)
            
            # Load performance history
            performance_path = os.path.join(directory, 'performance_history.pkl')
            if os.path.exists(performance_path):
                with open(performance_path, 'rb') as f:
                    self.performance_history = pickle.load(f)
            
            self.is_initialized = True
            logger.info(f"📂 Intelligence system loaded from {directory}")
            
        except Exception as e:
            logger.error(f"Error loading intelligence system: {e}")


async def main():
    """Test advanced intelligence system"""
    # Create sample data
    symbols = ['BTCUSDT', 'ETHUSDT']
    historical_data = {}
    
    for symbol in symbols:
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1H')
        prices = 45000 + np.cumsum(np.random.randn(1000) * 100)
        
        historical_data[symbol] = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'open': prices + np.random.randn(1000) * 50,
            'volume': np.random.lognormal(15, 1, 1000)
        })
    
    # Initialize intelligence system
    intelligence_system = AdvancedIntelligenceSystem(symbols)
    await intelligence_system.initialize_system(historical_data)
    
    # Test enhanced prediction
    test_data = historical_data['BTCUSDT'].tail(100)
    base_predictions = {
        'enhanced_transformer': 0.75,
        'cnn_lstm': 0.68,
        'technical_indicators': 0.72,
        'smc_analyzer': 0.71
    }
    base_confidences = {
        'enhanced_transformer': 0.85,
        'cnn_lstm': 0.78,
        'technical_indicators': 0.65,
        'smc_analyzer': 0.82
    }
    
    enhanced_result = await intelligence_system.get_enhanced_prediction(
        'BTCUSDT', test_data, base_predictions, base_confidences
    )
    
    print("🧠 Advanced Intelligence System Results:")
    print(f"Enhanced Prediction: {enhanced_result['prediction']:.3f}")
    print(f"Enhanced Confidence: {enhanced_result['confidence']:.3f}")
    print(f"Intelligence Score: {enhanced_result['intelligence_metrics']['intelligence_score']:.3f}")
    print(f"Enhancement Factor: {enhanced_result['intelligence_metrics']['enhancement_factor']:.2f}x")
    
    # Get system summary
    summary = intelligence_system.get_intelligence_summary()
    print(f"\nSystem Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
