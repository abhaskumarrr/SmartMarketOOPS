#!/usr/bin/env python3
"""
Live Performance Validation System for Enhanced SmartMarketOOPS
Validates system performance with real market data and automatic parameter adjustment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path

from ..trading.multi_symbol_manager import MultiSymbolTradingManager
from ..risk.advanced_risk_manager import AdvancedRiskManager
from ..data.real_market_data_service import get_market_data_service
from ..api.enhanced_model_service import EnhancedModelService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Live performance metrics"""
    timestamp: datetime
    symbol: str
    prediction: float
    actual_outcome: float
    confidence: float
    quality_score: float
    trade_executed: bool
    pnl: Optional[float] = None
    accuracy: Optional[float] = None
    signal_valid: bool = True


@dataclass
class ValidationResults:
    """Validation results summary"""
    total_predictions: int
    accurate_predictions: int
    accuracy_rate: float
    avg_confidence: float
    avg_quality_score: float
    total_trades: int
    profitable_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float


class LivePerformanceValidator:
    """Live performance validation with automatic parameter adjustment"""
    
    def __init__(self):
        """Initialize the live performance validator"""
        self.multi_symbol_manager = MultiSymbolTradingManager()
        self.risk_manager = AdvancedRiskManager()
        self.enhanced_service = EnhancedModelService()
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.validation_results = {}
        self.parameter_adjustments = []
        
        # Validation settings
        self.validation_window = 100  # Number of predictions to evaluate
        self.adjustment_threshold = 0.05  # 5% performance change threshold
        self.min_confidence_threshold = 0.6
        self.max_confidence_threshold = 0.9
        
        # Real-time tracking
        self.is_validating = False
        self.validation_start_time = None
        
        logger.info("Live Performance Validator initialized")
    
    async def start_live_validation(self, duration_hours: int = 24):
        """Start live performance validation"""
        logger.info(f"🚀 Starting live performance validation for {duration_hours} hours")
        
        self.is_validating = True
        self.validation_start_time = datetime.now()
        
        # Initialize components
        await self.multi_symbol_manager.initialize()
        
        try:
            end_time = datetime.now() + timedelta(hours=duration_hours)
            
            while self.is_validating and datetime.now() < end_time:
                # Run validation cycle
                await self._run_validation_cycle()
                
                # Check if parameter adjustment is needed
                if len(self.performance_history) % 50 == 0:  # Every 50 predictions
                    await self._evaluate_and_adjust_parameters()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30-second intervals
                
        except Exception as e:
            logger.error(f"Error in live validation: {e}")
        finally:
            self.is_validating = False
            await self._generate_final_validation_report()
    
    async def _run_validation_cycle(self):
        """Run one validation cycle"""
        try:
            # Generate signals for all symbols
            signals = await self.multi_symbol_manager.generate_multi_symbol_signals()
            
            # Process each signal
            for symbol, signal_data in signals.items():
                await self._validate_signal(symbol, signal_data)
            
            # Monitor existing positions
            closed_positions = await self.multi_symbol_manager.monitor_positions()
            
            # Update performance metrics for closed positions
            for position in closed_positions:
                await self._update_performance_from_trade(position)
            
        except Exception as e:
            logger.error(f"Error in validation cycle: {e}")
    
    async def _validate_signal(self, symbol: str, signal_data: Dict[str, Any]):
        """Validate a single signal prediction"""
        try:
            prediction_data = signal_data['prediction']
            
            # Store prediction for later validation
            performance_metric = PerformanceMetrics(
                timestamp=datetime.now(),
                symbol=symbol,
                prediction=prediction_data.get('prediction', 0.5),
                actual_outcome=None,  # Will be updated later
                confidence=prediction_data.get('confidence', 0.0),
                quality_score=prediction_data.get('quality_score', 0.0),
                trade_executed=False,
                signal_valid=prediction_data.get('signal_valid', False)
            )
            
            # Execute trade if signal is strong enough
            if self._should_execute_trade(prediction_data):
                trades = await self.multi_symbol_manager.execute_multi_symbol_trades({symbol: signal_data})
                performance_metric.trade_executed = len(trades) > 0
            
            self.performance_history.append(performance_metric)
            
            # Keep only recent history
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error validating signal for {symbol}: {e}")
    
    def _should_execute_trade(self, prediction_data: Dict[str, Any]) -> bool:
        """Determine if a trade should be executed based on current thresholds"""
        
        if not prediction_data.get('signal_valid', False):
            return False
        
        confidence = prediction_data.get('confidence', 0)
        quality_score = prediction_data.get('quality_score', 0)
        
        # Use current dynamic thresholds
        current_thresholds = self._get_current_thresholds()
        
        return (confidence >= current_thresholds['confidence'] and 
                quality_score >= current_thresholds['quality'])
    
    def _get_current_thresholds(self) -> Dict[str, float]:
        """Get current dynamic thresholds"""
        # Start with default thresholds
        thresholds = {
            'confidence': 0.7,
            'quality': 0.6
        }
        
        # Apply any recent adjustments
        if self.parameter_adjustments:
            latest_adjustment = self.parameter_adjustments[-1]
            thresholds.update(latest_adjustment.get('new_thresholds', {}))
        
        return thresholds
    
    async def _update_performance_from_trade(self, closed_position: Dict[str, Any]):
        """Update performance metrics from a closed trade"""
        try:
            symbol = closed_position['symbol']
            entry_time = closed_position['entry_time']
            pnl_pct = closed_position.get('pnl_pct', 0)
            
            # Find corresponding prediction
            for metric in reversed(self.performance_history):
                if (metric.symbol == symbol and 
                    metric.trade_executed and 
                    abs((metric.timestamp - entry_time).total_seconds()) < 300):  # Within 5 minutes
                    
                    # Update with actual outcome
                    metric.actual_outcome = 1.0 if pnl_pct > 0 else 0.0
                    metric.pnl = pnl_pct
                    
                    # Calculate accuracy
                    predicted_direction = 1 if metric.prediction > 0.5 else 0
                    actual_direction = 1 if pnl_pct > 0 else 0
                    metric.accuracy = 1.0 if predicted_direction == actual_direction else 0.0
                    
                    logger.info(f"📊 Performance updated for {symbol}: "
                               f"Pred={metric.prediction:.3f}, "
                               f"Actual={'WIN' if pnl_pct > 0 else 'LOSS'}, "
                               f"P&L={pnl_pct:.2f}%")
                    break
            
        except Exception as e:
            logger.error(f"Error updating performance from trade: {e}")
    
    async def _evaluate_and_adjust_parameters(self):
        """Evaluate performance and adjust parameters if needed"""
        try:
            logger.info("🔍 Evaluating performance for parameter adjustment...")
            
            # Calculate recent performance
            recent_metrics = self._get_recent_metrics(window=50)
            
            if len(recent_metrics) < 20:  # Need sufficient data
                return
            
            # Calculate performance metrics
            validation_results = self._calculate_validation_results(recent_metrics)
            
            # Determine if adjustment is needed
            adjustment_needed, adjustment_type = self._should_adjust_parameters(validation_results)
            
            if adjustment_needed:
                new_thresholds = await self._calculate_new_thresholds(validation_results, adjustment_type)
                await self._apply_parameter_adjustment(new_thresholds, adjustment_type)
            
        except Exception as e:
            logger.error(f"Error in parameter evaluation: {e}")
    
    def _get_recent_metrics(self, window: int = 50) -> List[PerformanceMetrics]:
        """Get recent performance metrics with actual outcomes"""
        recent_metrics = []
        
        for metric in reversed(self.performance_history):
            if metric.actual_outcome is not None:
                recent_metrics.append(metric)
                if len(recent_metrics) >= window:
                    break
        
        return list(reversed(recent_metrics))
    
    def _calculate_validation_results(self, metrics: List[PerformanceMetrics]) -> ValidationResults:
        """Calculate validation results from metrics"""
        if not metrics:
            return ValidationResults(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        total_predictions = len(metrics)
        accurate_predictions = sum(1 for m in metrics if m.accuracy == 1.0)
        accuracy_rate = accurate_predictions / total_predictions
        
        avg_confidence = np.mean([m.confidence for m in metrics])
        avg_quality_score = np.mean([m.quality_score for m in metrics])
        
        traded_metrics = [m for m in metrics if m.trade_executed]
        total_trades = len(traded_metrics)
        profitable_trades = sum(1 for m in traded_metrics if m.pnl and m.pnl > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(m.pnl for m in traded_metrics if m.pnl) if traded_metrics else 0
        
        # Calculate Sharpe ratio (simplified)
        if traded_metrics:
            returns = [m.pnl / 100 for m in traded_metrics if m.pnl]
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown (simplified)
        if traded_metrics:
            cumulative_pnl = np.cumsum([m.pnl for m in traded_metrics if m.pnl])
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = cumulative_pnl - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        return ValidationResults(
            total_predictions=total_predictions,
            accurate_predictions=accurate_predictions,
            accuracy_rate=accuracy_rate,
            avg_confidence=avg_confidence,
            avg_quality_score=avg_quality_score,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
    
    def _should_adjust_parameters(self, results: ValidationResults) -> tuple[bool, str]:
        """Determine if parameter adjustment is needed"""
        
        # Check if performance is below targets
        if results.accuracy_rate < 0.6:  # Below 60% accuracy
            return True, "INCREASE_SELECTIVITY"
        
        if results.win_rate < 0.5 and results.total_trades > 10:  # Below 50% win rate
            return True, "INCREASE_SELECTIVITY"
        
        # Check if we're being too selective
        if results.total_trades < 5 and results.total_predictions > 30:  # Too few trades
            return True, "DECREASE_SELECTIVITY"
        
        # Check if confidence is consistently high but accuracy is low
        if results.avg_confidence > 0.8 and results.accuracy_rate < 0.65:
            return True, "RECALIBRATE_CONFIDENCE"
        
        return False, ""
    
    async def _calculate_new_thresholds(self, results: ValidationResults, adjustment_type: str) -> Dict[str, float]:
        """Calculate new parameter thresholds"""
        current_thresholds = self._get_current_thresholds()
        new_thresholds = current_thresholds.copy()
        
        if adjustment_type == "INCREASE_SELECTIVITY":
            # Increase thresholds to be more selective
            new_thresholds['confidence'] = min(
                current_thresholds['confidence'] + 0.05,
                self.max_confidence_threshold
            )
            new_thresholds['quality'] = min(
                current_thresholds['quality'] + 0.05,
                0.8
            )
            
        elif adjustment_type == "DECREASE_SELECTIVITY":
            # Decrease thresholds to be less selective
            new_thresholds['confidence'] = max(
                current_thresholds['confidence'] - 0.05,
                self.min_confidence_threshold
            )
            new_thresholds['quality'] = max(
                current_thresholds['quality'] - 0.05,
                0.4
            )
            
        elif adjustment_type == "RECALIBRATE_CONFIDENCE":
            # Adjust based on confidence vs accuracy relationship
            confidence_accuracy_ratio = results.avg_confidence / max(results.accuracy_rate, 0.1)
            if confidence_accuracy_ratio > 1.2:  # Overconfident
                new_thresholds['confidence'] = min(
                    current_thresholds['confidence'] + 0.1,
                    self.max_confidence_threshold
                )
        
        return new_thresholds
    
    async def _apply_parameter_adjustment(self, new_thresholds: Dict[str, float], adjustment_type: str):
        """Apply parameter adjustments to the system"""
        try:
            # Record the adjustment
            adjustment_record = {
                'timestamp': datetime.now().isoformat(),
                'adjustment_type': adjustment_type,
                'old_thresholds': self._get_current_thresholds(),
                'new_thresholds': new_thresholds,
                'reason': f"Performance-based adjustment: {adjustment_type}"
            }
            
            self.parameter_adjustments.append(adjustment_record)
            
            # Update symbol configurations
            for symbol, config in self.multi_symbol_manager.symbol_configs.items():
                config.confidence_threshold = new_thresholds.get('confidence', config.confidence_threshold)
                config.quality_threshold = new_thresholds.get('quality', config.quality_threshold)
            
            logger.info(f"🔧 Parameter adjustment applied: {adjustment_type}")
            logger.info(f"   New confidence threshold: {new_thresholds.get('confidence', 'unchanged'):.3f}")
            logger.info(f"   New quality threshold: {new_thresholds.get('quality', 'unchanged'):.3f}")
            
        except Exception as e:
            logger.error(f"Error applying parameter adjustment: {e}")
    
    async def _generate_final_validation_report(self):
        """Generate final validation report"""
        logger.info("\n" + "="*80)
        logger.info("LIVE PERFORMANCE VALIDATION REPORT")
        logger.info("="*80)
        
        # Calculate overall results
        all_metrics = [m for m in self.performance_history if m.actual_outcome is not None]
        overall_results = self._calculate_validation_results(all_metrics)
        
        logger.info(f"Validation Period: {self.validation_start_time} to {datetime.now()}")
        logger.info(f"Total Duration: {datetime.now() - self.validation_start_time}")
        logger.info("")
        
        logger.info("PREDICTION PERFORMANCE:")
        logger.info(f"  Total Predictions: {overall_results.total_predictions}")
        logger.info(f"  Accurate Predictions: {overall_results.accurate_predictions}")
        logger.info(f"  Accuracy Rate: {overall_results.accuracy_rate:.1%}")
        logger.info(f"  Average Confidence: {overall_results.avg_confidence:.3f}")
        logger.info(f"  Average Quality Score: {overall_results.avg_quality_score:.3f}")
        logger.info("")
        
        logger.info("TRADING PERFORMANCE:")
        logger.info(f"  Total Trades: {overall_results.total_trades}")
        logger.info(f"  Profitable Trades: {overall_results.profitable_trades}")
        logger.info(f"  Win Rate: {overall_results.win_rate:.1%}")
        logger.info(f"  Total P&L: {overall_results.total_pnl:.2f}%")
        logger.info(f"  Sharpe Ratio: {overall_results.sharpe_ratio:.3f}")
        logger.info(f"  Max Drawdown: {overall_results.max_drawdown:.2f}%")
        logger.info("")
        
        logger.info("PARAMETER ADJUSTMENTS:")
        logger.info(f"  Total Adjustments: {len(self.parameter_adjustments)}")
        for i, adj in enumerate(self.parameter_adjustments[-3:], 1):  # Show last 3
            logger.info(f"  Adjustment {i}: {adj['adjustment_type']} at {adj['timestamp']}")
        
        # Save detailed report
        await self._save_validation_report(overall_results)
        
        logger.info("\n✅ Live performance validation completed")
    
    async def _save_validation_report(self, results: ValidationResults):
        """Save detailed validation report to file"""
        try:
            report_data = {
                'validation_summary': {
                    'start_time': self.validation_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_predictions': results.total_predictions,
                    'accuracy_rate': results.accuracy_rate,
                    'win_rate': results.win_rate,
                    'total_pnl': results.total_pnl,
                    'sharpe_ratio': results.sharpe_ratio
                },
                'parameter_adjustments': self.parameter_adjustments,
                'performance_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'symbol': m.symbol,
                        'prediction': m.prediction,
                        'confidence': m.confidence,
                        'quality_score': m.quality_score,
                        'actual_outcome': m.actual_outcome,
                        'pnl': m.pnl,
                        'accuracy': m.accuracy
                    }
                    for m in self.performance_history[-100:]  # Last 100 records
                ]
            }
            
            filename = f"live_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"📄 Detailed validation report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")


async def main():
    """Run live performance validation"""
    validator = LivePerformanceValidator()
    
    # Run validation for 2 hours (demo)
    await validator.start_live_validation(duration_hours=2)


if __name__ == "__main__":
    asyncio.run(main())
