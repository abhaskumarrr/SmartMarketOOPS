#!/usr/bin/env python3
"""
Parameter Optimization for Enhanced SmartMarketOOPS System
Optimizes ensemble weights, confidence thresholds, and market regime parameters
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Parameter optimizer for enhanced trading system"""
    
    def __init__(self):
        """Initialize the parameter optimizer"""
        self.current_params = {
            'ensemble_weights': {
                'enhanced_transformer': 0.4,
                'cnn_lstm': 0.3,
                'technical_indicators': 0.2,
                'smc_analyzer': 0.1
            },
            'confidence_threshold': 0.7,
            'quality_threshold': 0.6,
            'regime_thresholds': {
                'trending_threshold': 25,
                'volatile_threshold': 0.02,
                'ranging_threshold': 0.5
            },
            'adaptive_settings': {
                'performance_window': 100,
                'decay_factor': 0.95,
                'min_confidence': 0.6,
                'max_confidence': 0.9
            }
        }
        
        self.optimization_history = []
        self.best_params = None
        self.best_score = 0.0
        
        logger.info("Parameter Optimizer initialized")
    
    def simulate_performance(self, params: Dict[str, Any], num_samples: int = 100) -> float:
        """
        Simulate system performance with given parameters
        Returns a performance score (0-1, higher is better)
        """
        # Simulate ensemble predictions with different weights
        ensemble_weights = params['ensemble_weights']
        confidence_threshold = params['confidence_threshold']
        quality_threshold = params['quality_threshold']
        
        total_score = 0.0
        valid_signals = 0
        
        for _ in range(num_samples):
            # Simulate individual model predictions
            transformer_pred = np.random.uniform(0.3, 0.8)
            transformer_conf = np.random.uniform(0.6, 0.9)
            
            cnn_lstm_pred = np.random.uniform(0.2, 0.7)
            cnn_lstm_conf = np.random.uniform(0.5, 0.8)
            
            technical_pred = np.random.uniform(0.4, 0.6)
            technical_conf = np.random.uniform(0.4, 0.7)
            
            smc_pred = np.random.uniform(0.3, 0.7)
            smc_conf = np.random.uniform(0.5, 0.8)
            
            # Calculate ensemble prediction
            ensemble_pred = (
                transformer_pred * ensemble_weights['enhanced_transformer'] +
                cnn_lstm_pred * ensemble_weights['cnn_lstm'] +
                technical_pred * ensemble_weights['technical_indicators'] +
                smc_pred * ensemble_weights['smc_analyzer']
            )
            
            # Calculate ensemble confidence
            ensemble_conf = (
                transformer_conf * ensemble_weights['enhanced_transformer'] +
                cnn_lstm_conf * ensemble_weights['cnn_lstm'] +
                technical_conf * ensemble_weights['technical_indicators'] +
                smc_conf * ensemble_weights['smc_analyzer']
            )
            
            # Simulate quality score
            quality_score = np.random.uniform(0.4, 0.8)
            
            # Check if signal passes thresholds
            if ensemble_conf >= confidence_threshold and quality_score >= quality_threshold:
                valid_signals += 1
                
                # Simulate trading outcome based on prediction quality
                # Better predictions (closer to 0 or 1) have higher success probability
                prediction_strength = abs(ensemble_pred - 0.5) * 2
                success_probability = 0.5 + (prediction_strength * ensemble_conf * 0.3)
                
                if np.random.random() < success_probability:
                    # Successful trade
                    trade_score = prediction_strength * ensemble_conf * quality_score
                    total_score += trade_score
                else:
                    # Failed trade
                    total_score -= 0.1  # Small penalty for failed trades
        
        # Calculate final performance score
        if valid_signals > 0:
            avg_score = total_score / valid_signals
            signal_rate = valid_signals / num_samples
            
            # Balance between signal quality and signal frequency
            performance_score = avg_score * 0.7 + signal_rate * 0.3
        else:
            performance_score = 0.0
        
        return max(0.0, min(1.0, performance_score))
    
    def optimize_ensemble_weights(self, iterations: int = 50) -> Dict[str, float]:
        """Optimize ensemble model weights"""
        logger.info("Optimizing ensemble weights...")
        
        best_weights = self.current_params['ensemble_weights'].copy()
        best_score = self.simulate_performance(self.current_params)
        
        for i in range(iterations):
            # Generate random weight variations
            weights = {}
            total_weight = 0
            
            # Generate random weights
            for model in best_weights.keys():
                weights[model] = np.random.uniform(0.05, 0.6)
                total_weight += weights[model]
            
            # Normalize weights to sum to 1
            for model in weights.keys():
                weights[model] /= total_weight
            
            # Test new weights
            test_params = self.current_params.copy()
            test_params['ensemble_weights'] = weights
            
            score = self.simulate_performance(test_params)
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                logger.info(f"Iteration {i+1}: New best score {score:.4f}")
        
        logger.info(f"Ensemble weight optimization completed. Best score: {best_score:.4f}")
        return best_weights
    
    def optimize_confidence_threshold(self, min_threshold: float = 0.5, max_threshold: float = 0.9, 
                                    steps: int = 20) -> float:
        """Optimize confidence threshold"""
        logger.info("Optimizing confidence threshold...")
        
        best_threshold = self.current_params['confidence_threshold']
        best_score = self.simulate_performance(self.current_params)
        
        thresholds = np.linspace(min_threshold, max_threshold, steps)
        
        for threshold in thresholds:
            test_params = self.current_params.copy()
            test_params['confidence_threshold'] = threshold
            
            score = self.simulate_performance(test_params)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Confidence threshold optimization completed. Best threshold: {best_threshold:.3f}")
        return best_threshold
    
    def optimize_quality_threshold(self, min_threshold: float = 0.4, max_threshold: float = 0.8, 
                                 steps: int = 20) -> float:
        """Optimize quality threshold"""
        logger.info("Optimizing quality threshold...")
        
        best_threshold = self.current_params['quality_threshold']
        best_score = self.simulate_performance(self.current_params)
        
        thresholds = np.linspace(min_threshold, max_threshold, steps)
        
        for threshold in thresholds:
            test_params = self.current_params.copy()
            test_params['quality_threshold'] = threshold
            
            score = self.simulate_performance(test_params)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Quality threshold optimization completed. Best threshold: {best_threshold:.3f}")
        return best_threshold
    
    def optimize_regime_parameters(self) -> Dict[str, float]:
        """Optimize market regime detection parameters"""
        logger.info("Optimizing market regime parameters...")
        
        best_params = self.current_params['regime_thresholds'].copy()
        best_score = self.simulate_performance(self.current_params)
        
        # Parameter ranges for optimization
        param_ranges = {
            'trending_threshold': (15, 35),
            'volatile_threshold': (0.01, 0.05),
            'ranging_threshold': (0.3, 0.7)
        }
        
        iterations = 30
        
        for i in range(iterations):
            # Generate random parameter variations
            test_regime_params = {}
            
            for param, (min_val, max_val) in param_ranges.items():
                test_regime_params[param] = np.random.uniform(min_val, max_val)
            
            # Test new parameters
            test_params = self.current_params.copy()
            test_params['regime_thresholds'] = test_regime_params
            
            score = self.simulate_performance(test_params)
            
            if score > best_score:
                best_score = score
                best_params = test_regime_params.copy()
                logger.info(f"Regime optimization iteration {i+1}: New best score {score:.4f}")
        
        logger.info(f"Regime parameter optimization completed. Best score: {best_score:.4f}")
        return best_params
    
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run comprehensive parameter optimization"""
        logger.info("Starting comprehensive parameter optimization...")
        
        # Record initial performance
        initial_score = self.simulate_performance(self.current_params)
        logger.info(f"Initial performance score: {initial_score:.4f}")
        
        # Optimize each component
        optimized_params = self.current_params.copy()
        
        # 1. Optimize ensemble weights
        optimized_params['ensemble_weights'] = self.optimize_ensemble_weights()
        
        # 2. Optimize confidence threshold
        optimized_params['confidence_threshold'] = self.optimize_confidence_threshold()
        
        # 3. Optimize quality threshold
        optimized_params['quality_threshold'] = self.optimize_quality_threshold()
        
        # 4. Optimize regime parameters
        optimized_params['regime_thresholds'] = self.optimize_regime_parameters()
        
        # Calculate final performance
        final_score = self.simulate_performance(optimized_params)
        improvement = ((final_score - initial_score) / initial_score) * 100 if initial_score > 0 else 0
        
        # Store optimization results
        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'initial_params': self.current_params,
            'optimized_params': optimized_params,
            'initial_score': initial_score,
            'final_score': final_score,
            'improvement_percent': improvement
        }
        
        self.optimization_history.append(optimization_result)
        
        if final_score > self.best_score:
            self.best_score = final_score
            self.best_params = optimized_params.copy()
        
        logger.info(f"Comprehensive optimization completed!")
        logger.info(f"Performance improvement: {improvement:.2f}%")
        
        return optimization_result
    
    def generate_optimization_report(self, optimization_result: Dict[str, Any]) -> str:
        """Generate optimization report"""
        report = []
        report.append("="*80)
        report.append("PARAMETER OPTIMIZATION REPORT")
        report.append("="*80)
        report.append(f"Optimization Date: {optimization_result['timestamp']}")
        report.append("")
        
        # Performance Summary
        report.append("PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        report.append(f"Initial Score:     {optimization_result['initial_score']:.4f}")
        report.append(f"Optimized Score:   {optimization_result['final_score']:.4f}")
        report.append(f"Improvement:       {optimization_result['improvement_percent']:+.2f}%")
        report.append("")
        
        # Parameter Changes
        initial = optimization_result['initial_params']
        optimized = optimization_result['optimized_params']
        
        report.append("ENSEMBLE WEIGHTS:")
        report.append("-" * 40)
        for model in initial['ensemble_weights']:
            old_weight = initial['ensemble_weights'][model]
            new_weight = optimized['ensemble_weights'][model]
            change = ((new_weight - old_weight) / old_weight) * 100 if old_weight > 0 else 0
            report.append(f"{model:20}: {old_weight:.3f} → {new_weight:.3f} ({change:+.1f}%)")
        report.append("")
        
        report.append("THRESHOLDS:")
        report.append("-" * 40)
        
        # Confidence threshold
        old_conf = initial['confidence_threshold']
        new_conf = optimized['confidence_threshold']
        conf_change = ((new_conf - old_conf) / old_conf) * 100 if old_conf > 0 else 0
        report.append(f"Confidence:        {old_conf:.3f} → {new_conf:.3f} ({conf_change:+.1f}%)")
        
        # Quality threshold
        old_qual = initial['quality_threshold']
        new_qual = optimized['quality_threshold']
        qual_change = ((new_qual - old_qual) / old_qual) * 100 if old_qual > 0 else 0
        report.append(f"Quality:           {old_qual:.3f} → {new_qual:.3f} ({qual_change:+.1f}%)")
        report.append("")
        
        report.append("REGIME PARAMETERS:")
        report.append("-" * 40)
        for param in initial['regime_thresholds']:
            old_val = initial['regime_thresholds'][param]
            new_val = optimized['regime_thresholds'][param]
            change = ((new_val - old_val) / old_val) * 100 if old_val > 0 else 0
            report.append(f"{param:18}: {old_val:.3f} → {new_val:.3f} ({change:+.1f}%)")
        
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if optimization_result['improvement_percent'] > 5:
            report.append("✅ Significant improvement achieved - deploy optimized parameters")
        elif optimization_result['improvement_percent'] > 0:
            report.append("⚠️  Minor improvement - consider A/B testing before deployment")
        else:
            report.append("❌ No improvement - keep current parameters")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_optimized_parameters(self, filepath: str):
        """Save optimized parameters to file"""
        if self.best_params:
            with open(filepath, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            logger.info(f"Optimized parameters saved to {filepath}")
        else:
            logger.warning("No optimized parameters to save")


def main():
    """Main function for parameter optimization"""
    print("⚙️ Enhanced SmartMarketOOPS Parameter Optimization")
    print("="*60)
    
    # Initialize optimizer
    optimizer = ParameterOptimizer()
    
    # Run comprehensive optimization
    result = optimizer.run_comprehensive_optimization()
    
    # Generate and display report
    report = optimizer.generate_optimization_report(result)
    print(report)
    
    # Save optimized parameters
    optimizer.save_optimized_parameters("optimized_parameters.json")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
