#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline for Enhanced SmartMarketOOPS
Implements continuous learning with performance feedback and market adaptation
"""

import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch

from ..models.model_factory import ModelFactory
from ..models.model_registry import ModelRegistry
from ..data.real_market_data_service import get_market_data_service
from ..data.transformer_preprocessor import TransformerPreprocessor
from ..validation.live_performance_validator import LivePerformanceValidator
from ..utils.config import MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrainingTrigger:
    """Conditions that trigger model retraining"""
    performance_degradation: bool = False
    data_drift_detected: bool = False
    scheduled_retrain: bool = False
    market_regime_change: bool = False
    manual_trigger: bool = False
    
    def should_retrain(self) -> bool:
        """Check if any trigger condition is met"""
        return any([
            self.performance_degradation,
            self.data_drift_detected,
            self.scheduled_retrain,
            self.market_regime_change,
            self.manual_trigger
        ])


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""
    min_performance_threshold: float = 0.6  # 60% accuracy threshold
    performance_window: int = 100  # Number of predictions to evaluate
    max_performance_degradation: float = 0.1  # 10% degradation threshold
    min_data_points: int = 1000  # Minimum data points for retraining
    retrain_frequency_hours: int = 24  # Scheduled retraining frequency
    data_drift_threshold: float = 0.15  # Data drift detection threshold
    backup_models_count: int = 3  # Number of backup models to keep


class AutomatedRetrainingPipeline:
    """Automated model retraining pipeline with continuous learning"""
    
    def __init__(self):
        """Initialize the automated retraining pipeline"""
        self.model_registry = ModelRegistry()
        self.market_data_service = None
        self.performance_validator = LivePerformanceValidator()
        
        # Configuration
        self.config = RetrainingConfig()
        self.supported_symbols = MODEL_CONFIG.get('supported_symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT'])
        
        # State tracking
        self.last_retrain_times = {}
        self.performance_history = {}
        self.data_drift_scores = {}
        self.is_retraining = {}
        
        # Retraining queue
        self.retrain_queue = asyncio.Queue()
        self.is_running = False
        
        logger.info("Automated Retraining Pipeline initialized")
    
    async def start_pipeline(self):
        """Start the automated retraining pipeline"""
        logger.info("🚀 Starting Automated Retraining Pipeline")
        
        self.is_running = True
        
        # Initialize market data service
        self.market_data_service = await get_market_data_service()
        
        # Initialize performance tracking for all symbols
        for symbol in self.supported_symbols:
            self.last_retrain_times[symbol] = datetime.now()
            self.performance_history[symbol] = []
            self.data_drift_scores[symbol] = 0.0
            self.is_retraining[symbol] = False
        
        # Schedule periodic checks
        schedule.every(1).hours.do(self._schedule_performance_check)
        schedule.every(6).hours.do(self._schedule_data_drift_check)
        schedule.every(24).hours.do(self._schedule_routine_retrain)
        
        # Start background tasks
        asyncio.create_task(self._run_scheduler())
        asyncio.create_task(self._process_retrain_queue())
        
        logger.info("✅ Automated Retraining Pipeline started")
    
    async def stop_pipeline(self):
        """Stop the automated retraining pipeline"""
        logger.info("Stopping Automated Retraining Pipeline...")
        self.is_running = False
        
        # Wait for any ongoing retraining to complete
        for symbol in self.supported_symbols:
            while self.is_retraining.get(symbol, False):
                await asyncio.sleep(1)
        
        logger.info("✅ Automated Retraining Pipeline stopped")
    
    async def _run_scheduler(self):
        """Run the scheduled tasks"""
        while self.is_running:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute
    
    def _schedule_performance_check(self):
        """Schedule performance check for all symbols"""
        asyncio.create_task(self._check_performance_degradation())
    
    def _schedule_data_drift_check(self):
        """Schedule data drift check for all symbols"""
        asyncio.create_task(self._check_data_drift())
    
    def _schedule_routine_retrain(self):
        """Schedule routine retraining for all symbols"""
        asyncio.create_task(self._schedule_routine_retraining())
    
    async def _check_performance_degradation(self):
        """Check for performance degradation across all symbols"""
        try:
            logger.info("🔍 Checking for performance degradation...")
            
            for symbol in self.supported_symbols:
                if self.is_retraining.get(symbol, False):
                    continue
                
                # Get recent performance metrics
                recent_performance = await self._get_recent_performance(symbol)
                
                if len(recent_performance) < self.config.performance_window:
                    continue
                
                # Calculate current performance
                current_accuracy = np.mean([p.get('accuracy', 0) for p in recent_performance])
                
                # Get baseline performance (from model metadata)
                baseline_accuracy = await self._get_baseline_performance(symbol)
                
                # Check for degradation
                if baseline_accuracy > 0:
                    degradation = (baseline_accuracy - current_accuracy) / baseline_accuracy
                    
                    if degradation > self.config.max_performance_degradation:
                        logger.warning(f"⚠️  Performance degradation detected for {symbol}: "
                                     f"{degradation:.1%} drop (current: {current_accuracy:.1%}, "
                                     f"baseline: {baseline_accuracy:.1%})")
                        
                        trigger = RetrainingTrigger(performance_degradation=True)
                        await self._queue_retraining(symbol, trigger)
                
        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")
    
    async def _check_data_drift(self):
        """Check for data drift in market conditions"""
        try:
            logger.info("🔍 Checking for data drift...")
            
            for symbol in self.supported_symbols:
                if self.is_retraining.get(symbol, False):
                    continue
                
                # Get recent market data
                recent_data = await self._get_recent_market_data(symbol, days=7)
                historical_data = await self._get_historical_market_data(symbol, days=30)
                
                if recent_data.empty or historical_data.empty:
                    continue
                
                # Calculate data drift score
                drift_score = self._calculate_data_drift(recent_data, historical_data)
                self.data_drift_scores[symbol] = drift_score
                
                if drift_score > self.config.data_drift_threshold:
                    logger.warning(f"⚠️  Data drift detected for {symbol}: "
                                 f"drift score {drift_score:.3f} > threshold {self.config.data_drift_threshold}")
                    
                    trigger = RetrainingTrigger(data_drift_detected=True)
                    await self._queue_retraining(symbol, trigger)
                
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
    
    async def _schedule_routine_retraining(self):
        """Schedule routine retraining based on time intervals"""
        try:
            logger.info("🔍 Checking for scheduled retraining...")
            
            for symbol in self.supported_symbols:
                if self.is_retraining.get(symbol, False):
                    continue
                
                last_retrain = self.last_retrain_times.get(symbol, datetime.now() - timedelta(days=1))
                time_since_retrain = datetime.now() - last_retrain
                
                if time_since_retrain.total_seconds() > (self.config.retrain_frequency_hours * 3600):
                    logger.info(f"📅 Scheduled retraining due for {symbol} "
                               f"(last retrain: {time_since_retrain.total_seconds() / 3600:.1f} hours ago)")
                    
                    trigger = RetrainingTrigger(scheduled_retrain=True)
                    await self._queue_retraining(symbol, trigger)
                
        except Exception as e:
            logger.error(f"Error checking scheduled retraining: {e}")
    
    async def _queue_retraining(self, symbol: str, trigger: RetrainingTrigger):
        """Queue a symbol for retraining"""
        if not self.is_retraining.get(symbol, False):
            await self.retrain_queue.put((symbol, trigger))
            logger.info(f"📋 Queued {symbol} for retraining (trigger: {trigger})")
    
    async def _process_retrain_queue(self):
        """Process the retraining queue"""
        while self.is_running:
            try:
                # Get next item from queue (with timeout)
                symbol, trigger = await asyncio.wait_for(
                    self.retrain_queue.get(), 
                    timeout=10.0
                )
                
                # Execute retraining
                await self._execute_retraining(symbol, trigger)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing retrain queue: {e}")
    
    async def _execute_retraining(self, symbol: str, trigger: RetrainingTrigger):
        """Execute model retraining for a symbol"""
        try:
            logger.info(f"🔄 Starting retraining for {symbol}")
            self.is_retraining[symbol] = True
            
            # Backup current model
            await self._backup_current_model(symbol)
            
            # Collect training data
            training_data = await self._collect_training_data(symbol)
            
            if len(training_data) < self.config.min_data_points:
                logger.warning(f"⚠️  Insufficient data for {symbol} retraining: "
                             f"{len(training_data)} < {self.config.min_data_points}")
                return
            
            # Prepare data for training
            preprocessor = TransformerPreprocessor(
                sequence_length=100,
                forecast_horizon=1,
                scaling_method='standard',
                feature_engineering=True,
                multi_timeframe=True,
                attention_features=True
            )
            
            processed_data = preprocessor.fit_transform(training_data, target_column='close')
            
            # Create enhanced model
            model = ModelFactory.create_model(
                model_type='enhanced_transformer',
                input_dim=processed_data['num_features'],
                output_dim=1,
                seq_len=processed_data['sequence_length'],
                forecast_horizon=1,
                **MODEL_CONFIG['transformer']
            )
            
            # Train the model
            train_loader, val_loader = preprocessor.create_data_loaders(
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_val'],
                processed_data['y_val'],
                batch_size=MODEL_CONFIG['batch_size']
            )
            
            logger.info(f"🏋️  Training enhanced model for {symbol}...")
            history = model.fit_model(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=MODEL_CONFIG['epochs'],
                lr=MODEL_CONFIG['learning_rate'],
                early_stopping_patience=MODEL_CONFIG['patience']
            )
            
            # Validate new model performance
            validation_score = await self._validate_new_model(symbol, model, preprocessor)
            
            # Compare with current model performance
            current_performance = await self._get_current_model_performance(symbol)
            
            if validation_score > current_performance:
                # Deploy new model
                await self._deploy_new_model(symbol, model, preprocessor, history, trigger)
                logger.info(f"✅ Successfully retrained and deployed model for {symbol} "
                           f"(validation score: {validation_score:.3f} > {current_performance:.3f})")
            else:
                logger.warning(f"⚠️  New model for {symbol} did not improve performance "
                             f"(validation score: {validation_score:.3f} <= {current_performance:.3f})")
                # Keep current model
            
            # Update retraining timestamp
            self.last_retrain_times[symbol] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error retraining model for {symbol}: {e}")
        finally:
            self.is_retraining[symbol] = False
    
    async def _collect_training_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Collect recent market data for training"""
        try:
            # Get historical data from market data service
            df = await self.market_data_service.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                limit=days * 24  # Hourly data for specified days
            )
            
            # Add technical indicators and features
            df = self._add_technical_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting training data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # Simple moving averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Drop NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
            return df
    
    def _calculate_data_drift(self, recent_data: pd.DataFrame, historical_data: pd.DataFrame) -> float:
        """Calculate data drift score between recent and historical data"""
        try:
            # Use key features for drift detection
            features = ['close', 'volume', 'high_low_ratio', 'price_change']
            
            drift_scores = []
            
            for feature in features:
                if feature in recent_data.columns and feature in historical_data.columns:
                    # Calculate statistical differences
                    recent_mean = recent_data[feature].mean()
                    historical_mean = historical_data[feature].mean()
                    
                    recent_std = recent_data[feature].std()
                    historical_std = historical_data[feature].std()
                    
                    # Normalized difference in means
                    mean_diff = abs(recent_mean - historical_mean) / (historical_std + 1e-8)
                    
                    # Difference in standard deviations
                    std_diff = abs(recent_std - historical_std) / (historical_std + 1e-8)
                    
                    # Combined drift score for this feature
                    feature_drift = (mean_diff + std_diff) / 2
                    drift_scores.append(feature_drift)
            
            # Overall drift score
            return np.mean(drift_scores) if drift_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating data drift: {e}")
            return 0.0
    
    async def _get_recent_performance(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent performance metrics for a symbol"""
        # This would integrate with the performance validator
        # For now, return mock data
        return []
    
    async def _get_baseline_performance(self, symbol: str) -> float:
        """Get baseline performance for a symbol"""
        try:
            # Load model metadata to get baseline performance
            _, metadata = self.model_registry.load_model(symbol, return_metadata=True)
            return metadata.get('baseline_accuracy', 0.6)
        except:
            return 0.6  # Default baseline
    
    async def _get_recent_market_data(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get recent market data for drift detection"""
        try:
            return await self.market_data_service.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                limit=days * 24
            )
        except:
            return pd.DataFrame()
    
    async def _get_historical_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical market data for comparison"""
        try:
            end_date = datetime.now() - timedelta(days=7)  # Start from 7 days ago
            # This would need to be implemented to get historical data from a specific date range
            return await self.market_data_service.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                limit=days * 24
            )
        except:
            return pd.DataFrame()
    
    async def _backup_current_model(self, symbol: str):
        """Backup the current model before retraining"""
        try:
            # Create backup with timestamp
            backup_version = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Load current model
            model, metadata = self.model_registry.load_model(symbol, return_metadata=True)
            
            # Save as backup
            self.model_registry.save_model(
                model=model,
                symbol=symbol,
                version=backup_version,
                metadata={**metadata, 'backup': True}
            )
            
            logger.info(f"📦 Backed up current model for {symbol} as {backup_version}")
            
        except Exception as e:
            logger.error(f"Error backing up model for {symbol}: {e}")
    
    async def _validate_new_model(self, symbol: str, model: Any, preprocessor: TransformerPreprocessor) -> float:
        """Validate new model performance"""
        try:
            # Get validation data
            validation_data = await self._collect_training_data(symbol, days=7)
            
            if validation_data.empty:
                return 0.0
            
            # Process validation data
            processed_data = preprocessor.transform(validation_data)
            
            # Generate predictions
            predictions = model.predict(processed_data['X_test'])
            
            # Calculate accuracy (simplified)
            actual_directions = (processed_data['y_test'] > 0.5).astype(int)
            predicted_directions = (predictions > 0.5).astype(int)
            
            accuracy = np.mean(actual_directions == predicted_directions)
            return accuracy
            
        except Exception as e:
            logger.error(f"Error validating new model for {symbol}: {e}")
            return 0.0
    
    async def _get_current_model_performance(self, symbol: str) -> float:
        """Get current model performance"""
        try:
            _, metadata = self.model_registry.load_model(symbol, return_metadata=True)
            return metadata.get('validation_accuracy', 0.6)
        except:
            return 0.6  # Default performance
    
    async def _deploy_new_model(self, symbol: str, model: Any, preprocessor: TransformerPreprocessor, 
                               history: Dict[str, Any], trigger: RetrainingTrigger):
        """Deploy the new trained model"""
        try:
            # Create new version
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create metadata
            metadata = {
                'symbol': symbol,
                'model_type': 'enhanced_transformer',
                'version': version,
                'retrain_trigger': str(trigger),
                'training_history': history,
                'deployed_at': datetime.now().isoformat(),
                'validation_accuracy': history.get('best_val_loss', 0.0)
            }
            
            # Save new model
            self.model_registry.save_model(
                model=model,
                symbol=symbol,
                version=version,
                metadata=metadata,
                preprocessor=preprocessor
            )
            
            logger.info(f"🚀 Deployed new model for {symbol} (version: {version})")
            
        except Exception as e:
            logger.error(f"Error deploying new model for {symbol}: {e}")
    
    def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining status"""
        return {
            'is_running': self.is_running,
            'symbols_retraining': [symbol for symbol, status in self.is_retraining.items() if status],
            'last_retrain_times': {
                symbol: time.isoformat() for symbol, time in self.last_retrain_times.items()
            },
            'data_drift_scores': self.data_drift_scores,
            'queue_size': self.retrain_queue.qsize()
        }


async def main():
    """Test the automated retraining pipeline"""
    pipeline = AutomatedRetrainingPipeline()
    
    # Start pipeline
    await pipeline.start_pipeline()
    
    # Run for a while
    await asyncio.sleep(3600)  # 1 hour
    
    # Stop pipeline
    await pipeline.stop_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
