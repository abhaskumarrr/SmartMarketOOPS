"""
ML Trading Intelligence Orchestrator
Task #31: ML Trading Intelligence Integration
Advanced ML system that orchestrates all trading intelligence components
"""

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ..models.memory_efficient_transformer import MemoryEfficientTransformer
from ..ensemble.enhanced_signal_quality_system import EnhancedSignalQualitySystem, TradingSignal
from ..integration.transformer_ml_pipeline import TransformerMLPipeline
from ..data.transformer_preprocessor import TransformerPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class MLIntelligenceConfig:
    """Configuration for ML Trading Intelligence"""
    # Model configurations
    transformer_config: Dict[str, Any]
    ensemble_config: Dict[str, Any]
    signal_quality_config: Dict[str, Any]
    
    # Intelligence parameters
    confidence_threshold: float = 0.7
    quality_threshold: str = 'good'
    max_concurrent_predictions: int = 5
    prediction_timeout: float = 30.0
    
    # Performance targets
    target_accuracy: float = 0.75
    target_win_rate: float = 0.70
    target_latency_ms: float = 100.0
    
    # Memory management
    max_memory_usage_gb: float = 2.0
    cleanup_interval_minutes: int = 5
    max_cached_predictions: int = 1000


@dataclass
class IntelligenceMetrics:
    """Comprehensive intelligence performance metrics"""
    # Accuracy metrics
    overall_accuracy: float
    transformer_accuracy: float
    ensemble_accuracy: float
    signal_quality_accuracy: float
    
    # Performance metrics
    prediction_latency_ms: float
    throughput_predictions_per_second: float
    memory_usage_gb: float
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Model health
    model_confidence: float
    prediction_consistency: float
    error_rate: float
    
    # System metrics
    uptime_percentage: float
    last_update: datetime


class MLTradingIntelligence:
    """
    Advanced ML Trading Intelligence System
    Orchestrates all ML components for optimal trading decisions
    """
    
    def __init__(self, config: MLIntelligenceConfig):
        """Initialize the ML Trading Intelligence system"""
        self.config = config
        self.is_initialized = False
        self.is_running = False
        
        # Core ML components
        self.transformer_pipeline: Optional[TransformerMLPipeline] = None
        self.signal_quality_system: Optional[EnhancedSignalQualitySystem] = None
        self.preprocessor: Optional[TransformerPreprocessor] = None
        
        # Intelligence state
        self.active_predictions: Dict[str, asyncio.Task] = {}
        self.prediction_cache: Dict[str, Dict] = {}
        self.performance_history: List[IntelligenceMetrics] = []
        
        # Metrics tracking
        self.current_metrics = IntelligenceMetrics(
            overall_accuracy=0.0,
            transformer_accuracy=0.0,
            ensemble_accuracy=0.0,
            signal_quality_accuracy=0.0,
            prediction_latency_ms=0.0,
            throughput_predictions_per_second=0.0,
            memory_usage_gb=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            model_confidence=0.0,
            prediction_consistency=0.0,
            error_rate=0.0,
            uptime_percentage=100.0,
            last_update=datetime.now()
        )
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("ML Trading Intelligence initialized")
    
    async def initialize(self) -> bool:
        """Initialize all ML components"""
        try:
            logger.info("Initializing ML Trading Intelligence components...")
            
            # Initialize Transformer pipeline
            self.transformer_pipeline = TransformerMLPipeline(
                use_memory_efficient=True,
                enable_ensemble=True,
                **self.config.ensemble_config
            )
            
            # Initialize models with optimal configuration
            await self._initialize_models()
            
            # Initialize signal quality system
            self.signal_quality_system = EnhancedSignalQualitySystem(
                transformer_model=self.transformer_pipeline.transformer_model,
                ensemble_models=self.transformer_pipeline.legacy_models,
                **self.config.signal_quality_config
            )
            
            # Initialize preprocessor
            self.preprocessor = TransformerPreprocessor(
                sequence_length=100,
                forecast_horizon=1
            )
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("ML Trading Intelligence initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Trading Intelligence: {str(e)}")
            return False
    
    async def _initialize_models(self):
        """Initialize ML models with optimal configuration"""
        # Determine optimal model configuration based on available memory
        memory_config = self._get_optimal_memory_config()
        
        self.transformer_pipeline.initialize_models(
            input_dim=memory_config['input_dim'],
            output_dim=1,
            seq_len=memory_config['seq_len'],
            forecast_horizon=1
        )
        
        logger.info(f"Models initialized with config: {memory_config}")
    
    def _get_optimal_memory_config(self) -> Dict[str, int]:
        """Get optimal model configuration for available memory"""
        # Memory-efficient configuration for M2 MacBook Air 8GB
        return {
            'input_dim': 20,  # Rich feature set
            'seq_len': 100,   # Optimal sequence length
            'd_model': 256,   # Enhanced model size
            'num_layers': 6,  # Deep architecture
            'nhead': 8        # Multi-head attention
        }
    
    async def generate_trading_intelligence(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        additional_context: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive trading intelligence for a symbol
        
        Args:
            market_data: Recent market data
            symbol: Trading symbol
            additional_context: Additional context (news, sentiment, etc.)
            
        Returns:
            Comprehensive trading intelligence or None if failed
        """
        start_time = datetime.now()
        
        try:
            # Check if prediction is already in progress
            if symbol in self.active_predictions:
                logger.debug(f"Prediction already in progress for {symbol}")
                return None
            
            # Create prediction task
            prediction_task = asyncio.create_task(
                self._generate_prediction_async(market_data, symbol, additional_context)
            )
            
            self.active_predictions[symbol] = prediction_task
            
            try:
                # Wait for prediction with timeout
                intelligence = await asyncio.wait_for(
                    prediction_task,
                    timeout=self.config.prediction_timeout
                )
                
                # Update performance metrics
                latency = (datetime.now() - start_time).total_seconds() * 1000
                await self._update_performance_metrics(latency, True)
                
                return intelligence
                
            except asyncio.TimeoutError:
                logger.warning(f"Prediction timeout for {symbol}")
                prediction_task.cancel()
                await self._update_performance_metrics(0, False)
                return None
                
        except Exception as e:
            logger.error(f"Error generating trading intelligence for {symbol}: {str(e)}")
            await self._update_performance_metrics(0, False)
            return None
            
        finally:
            # Clean up active prediction
            self.active_predictions.pop(symbol, None)
    
    async def _generate_prediction_async(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        additional_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate prediction asynchronously"""
        
        # 1. Generate high-quality trading signal
        signal = self.signal_quality_system.generate_signal(
            market_data=market_data,
            symbol=symbol,
            current_price=market_data['close'].iloc[-1],
            smc_analysis=additional_context.get('smc_analysis') if additional_context else None,
            technical_indicators=additional_context.get('technical_indicators') if additional_context else None
        )
        
        if not signal:
            raise ValueError("Failed to generate trading signal")
        
        # 2. Get enhanced predictions from pipeline
        pipeline_prediction = self.transformer_pipeline.predict(
            market_data=market_data,
            symbol=symbol,
            return_signal=False,
            use_ensemble=True
        )
        
        # 3. Generate market regime analysis
        regime_analysis = await self._analyze_market_regime(market_data)
        
        # 4. Calculate risk assessment
        risk_assessment = await self._calculate_risk_assessment(signal, market_data)
        
        # 5. Generate execution strategy
        execution_strategy = await self._generate_execution_strategy(signal, regime_analysis)
        
        # 6. Compile comprehensive intelligence
        intelligence = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal': asdict(signal),
            'pipeline_prediction': pipeline_prediction.tolist() if isinstance(pipeline_prediction, np.ndarray) else pipeline_prediction,
            'regime_analysis': regime_analysis,
            'risk_assessment': risk_assessment,
            'execution_strategy': execution_strategy,
            'confidence_score': signal.confidence,
            'quality_rating': signal.quality,
            'intelligence_version': '1.0'
        }
        
        # 7. Cache the intelligence
        self._cache_intelligence(symbol, intelligence)
        
        return intelligence
    
    async def _analyze_market_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime"""
        try:
            # Calculate volatility regime
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            volatility_percentile = (returns.rolling(100).std() <= volatility).mean()
            
            # Calculate trend regime
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
            current_price = market_data['close'].iloc[-1]
            
            trend_strength = abs(current_price - sma_20) / sma_20
            trend_direction = 'bullish' if sma_20 > sma_50 else 'bearish'
            
            # Calculate volume regime
            volume_ma = market_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma
            
            # Determine overall regime
            if volatility_percentile > 0.8:
                volatility_regime = 'high'
            elif volatility_percentile > 0.5:
                volatility_regime = 'medium'
            else:
                volatility_regime = 'low'
            
            if trend_strength > 0.05:
                trend_regime = f'strong_{trend_direction}'
            elif trend_strength > 0.02:
                trend_regime = f'moderate_{trend_direction}'
            else:
                trend_regime = 'sideways'
            
            return {
                'volatility_regime': volatility_regime,
                'volatility_percentile': volatility_percentile,
                'trend_regime': trend_regime,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'volume_regime': 'high' if volume_ratio > 1.5 else 'normal' if volume_ratio > 0.8 else 'low',
                'volume_ratio': volume_ratio,
                'market_condition': self._determine_market_condition(volatility_regime, trend_regime)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {str(e)}")
            return {'error': str(e)}
    
    def _determine_market_condition(self, volatility_regime: str, trend_regime: str) -> str:
        """Determine overall market condition"""
        if 'strong' in trend_regime and volatility_regime == 'low':
            return 'trending_stable'
        elif 'strong' in trend_regime and volatility_regime == 'high':
            return 'trending_volatile'
        elif 'sideways' in trend_regime and volatility_regime == 'low':
            return 'consolidating'
        elif 'sideways' in trend_regime and volatility_regime == 'high':
            return 'choppy'
        else:
            return 'transitional'
    
    async def _calculate_risk_assessment(self, signal: TradingSignal, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment"""
        try:
            # Calculate Value at Risk (VaR)
            returns = market_data['close'].pct_change().dropna()
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Calculate maximum adverse excursion
            if signal.signal_type in ['buy', 'strong_buy']:
                mae = (market_data['low'].rolling(20).min().iloc[-1] - signal.price) / signal.price
            else:
                mae = (signal.price - market_data['high'].rolling(20).max().iloc[-1]) / signal.price
            
            # Calculate position sizing based on Kelly criterion
            win_rate = 0.7  # From performance metrics
            avg_win = 0.03  # 3% average win
            avg_loss = 0.015  # 1.5% average loss
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Risk-adjusted position size
            risk_adjusted_size = min(signal.position_size or 0.1, kelly_fraction * 0.5)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'maximum_adverse_excursion': mae,
                'kelly_fraction': kelly_fraction,
                'risk_adjusted_position_size': risk_adjusted_size,
                'risk_reward_ratio': signal.risk_reward_ratio or 2.0,
                'confidence_adjusted_risk': signal.confidence * risk_adjusted_size,
                'risk_level': self._categorize_risk_level(var_95, mae, signal.confidence)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk assessment: {str(e)}")
            return {'error': str(e)}
    
    def _categorize_risk_level(self, var_95: float, mae: float, confidence: float) -> str:
        """Categorize overall risk level"""
        risk_score = abs(var_95) * 0.4 + abs(mae) * 0.4 + (1 - confidence) * 0.2
        
        if risk_score < 0.02:
            return 'low'
        elif risk_score < 0.05:
            return 'medium'
        else:
            return 'high'
    
    async def _generate_execution_strategy(self, signal: TradingSignal, regime_analysis: Dict) -> Dict[str, Any]:
        """Generate optimal execution strategy"""
        try:
            market_condition = regime_analysis.get('market_condition', 'unknown')
            
            # Base execution parameters
            execution_strategy = {
                'entry_method': 'market',
                'exit_method': 'limit',
                'time_in_force': 'GTC',
                'execution_urgency': 'normal'
            }
            
            # Adjust based on market condition
            if market_condition == 'trending_stable':
                execution_strategy.update({
                    'entry_method': 'limit',
                    'entry_offset_pct': 0.001,  # 0.1% better than market
                    'execution_urgency': 'patient'
                })
            elif market_condition == 'trending_volatile':
                execution_strategy.update({
                    'entry_method': 'market',
                    'execution_urgency': 'urgent',
                    'partial_fill_allowed': True
                })
            elif market_condition == 'choppy':
                execution_strategy.update({
                    'entry_method': 'limit',
                    'entry_offset_pct': 0.002,  # 0.2% better than market
                    'execution_urgency': 'very_patient',
                    'time_in_force': 'IOC'  # Immediate or cancel
                })
            
            # Add signal-specific adjustments
            if signal.quality == 'excellent':
                execution_strategy['execution_urgency'] = 'urgent'
            elif signal.quality == 'poor':
                execution_strategy['execution_urgency'] = 'very_patient'
            
            # Add timing recommendations
            execution_strategy.update({
                'recommended_timing': self._get_optimal_timing(regime_analysis),
                'max_execution_time_minutes': self._get_max_execution_time(market_condition),
                'slippage_tolerance_pct': self._get_slippage_tolerance(market_condition)
            })
            
            return execution_strategy
            
        except Exception as e:
            logger.error(f"Error generating execution strategy: {str(e)}")
            return {'error': str(e)}
    
    def _get_optimal_timing(self, regime_analysis: Dict) -> str:
        """Get optimal timing for execution"""
        volume_regime = regime_analysis.get('volume_regime', 'normal')
        
        if volume_regime == 'high':
            return 'immediate'
        elif volume_regime == 'low':
            return 'wait_for_volume'
        else:
            return 'normal'
    
    def _get_max_execution_time(self, market_condition: str) -> int:
        """Get maximum execution time in minutes"""
        timing_map = {
            'trending_stable': 30,
            'trending_volatile': 5,
            'consolidating': 60,
            'choppy': 15,
            'transitional': 20
        }
        return timing_map.get(market_condition, 15)
    
    def _get_slippage_tolerance(self, market_condition: str) -> float:
        """Get slippage tolerance percentage"""
        slippage_map = {
            'trending_stable': 0.001,  # 0.1%
            'trending_volatile': 0.005,  # 0.5%
            'consolidating': 0.002,  # 0.2%
            'choppy': 0.003,  # 0.3%
            'transitional': 0.004  # 0.4%
        }
        return slippage_map.get(market_condition, 0.002)
    
    def _cache_intelligence(self, symbol: str, intelligence: Dict):
        """Cache intelligence with memory management"""
        cache_key = f"{symbol}_{intelligence['timestamp']}"
        self.prediction_cache[cache_key] = intelligence
        
        # Memory management - keep only recent predictions
        if len(self.prediction_cache) > self.config.max_cached_predictions:
            # Remove oldest predictions
            sorted_keys = sorted(self.prediction_cache.keys())
            keys_to_remove = sorted_keys[:-self.config.max_cached_predictions]
            for key in keys_to_remove:
                del self.prediction_cache[key]
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        self.background_tasks = [
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._memory_manager()),
            asyncio.create_task(self._model_health_checker())
        ]
        
        logger.info("Background tasks started")
    
    async def _performance_monitor(self):
        """Monitor system performance continuously"""
        while self.is_running:
            try:
                await self._update_system_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in performance monitor: {str(e)}")
                await asyncio.sleep(60)
    
    async def _memory_manager(self):
        """Manage memory usage and cleanup"""
        while self.is_running:
            try:
                await self._cleanup_memory()
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in memory manager: {str(e)}")
                await asyncio.sleep(300)  # 5 minutes
    
    async def _model_health_checker(self):
        """Monitor model health and performance"""
        while self.is_running:
            try:
                await self._check_model_health()
                await asyncio.sleep(120)  # Check every 2 minutes
            except Exception as e:
                logger.error(f"Error in model health checker: {str(e)}")
                await asyncio.sleep(300)
    
    async def _update_system_metrics(self):
        """Update comprehensive system metrics"""
        # This would be implemented with actual metric collection
        pass
    
    async def _cleanup_memory(self):
        """Perform memory cleanup"""
        # Clean up old cached predictions
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)
        
        keys_to_remove = []
        for key, intelligence in self.prediction_cache.items():
            timestamp_str = intelligence.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp < cutoff_time:
                    keys_to_remove.append(key)
            except:
                keys_to_remove.append(key)  # Remove invalid entries
        
        for key in keys_to_remove:
            del self.prediction_cache[key]
        
        logger.debug(f"Cleaned up {len(keys_to_remove)} cached predictions")
    
    async def _check_model_health(self):
        """Check model health and performance"""
        # This would implement model health checks
        pass
    
    async def _update_performance_metrics(self, latency: float, success: bool):
        """Update performance metrics"""
        # Update latency
        if success:
            self.current_metrics.prediction_latency_ms = latency
        
        # Update error rate
        if not success:
            self.current_metrics.error_rate += 0.01
        else:
            self.current_metrics.error_rate = max(0, self.current_metrics.error_rate - 0.001)
        
        self.current_metrics.last_update = datetime.now()
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get comprehensive intelligence system summary"""
        return {
            'system_status': {
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'active_predictions': len(self.active_predictions),
                'cached_predictions': len(self.prediction_cache)
            },
            'performance_metrics': asdict(self.current_metrics),
            'configuration': asdict(self.config),
            'component_status': {
                'transformer_pipeline': self.transformer_pipeline is not None,
                'signal_quality_system': self.signal_quality_system is not None,
                'preprocessor': self.preprocessor is not None
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the intelligence system"""
        logger.info("Shutting down ML Trading Intelligence...")
        
        self.is_running = False
        
        # Cancel active predictions
        for task in self.active_predictions.values():
            task.cancel()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("ML Trading Intelligence shutdown completed")
