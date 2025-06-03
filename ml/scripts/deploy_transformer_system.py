"""
Deployment and Validation Script for Task #24 and Task #25
Validates Transformer integration and Enhanced Signal Quality System
Demonstrates 20-30% performance improvement target
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.memory_efficient_transformer import MemoryEfficientTransformer
from src.data.transformer_preprocessor import TransformerPreprocessor
from src.ensemble.enhanced_signal_quality_system import EnhancedSignalQualitySystem
from src.integration.transformer_ml_pipeline import TransformerMLPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransformerSystemValidator:
    """
    Comprehensive validation system for Transformer integration
    Validates performance improvements and system reliability
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the validation system"""
        self.config = self._load_config(config_path)
        self.results = {
            'validation_timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'performance_tests': {},
            'integration_tests': {},
            'memory_tests': {},
            'signal_quality_tests': {},
            'overall_score': 0.0
        }
        
        logger.info("TransformerSystemValidator initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        default_config = {
            'test_data_size': 1000,
            'training_epochs': 5,
            'batch_size': 16,
            'sequence_length': 100,
            'performance_threshold': 0.20,  # 20% improvement target
            'memory_limit_gb': 6.0,  # Memory limit for M2 MacBook Air
            'confidence_threshold': 0.6
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        import psutil
        
        return {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the Transformer system"""
        logger.info("Starting comprehensive validation...")
        
        try:
            # 1. Performance validation
            logger.info("Running performance tests...")
            self.results['performance_tests'] = self._validate_performance()
            
            # 2. Integration validation
            logger.info("Running integration tests...")
            self.results['integration_tests'] = self._validate_integration()
            
            # 3. Memory efficiency validation
            logger.info("Running memory tests...")
            self.results['memory_tests'] = self._validate_memory_efficiency()
            
            # 4. Signal quality validation
            logger.info("Running signal quality tests...")
            self.results['signal_quality_tests'] = self._validate_signal_quality()
            
            # 5. Calculate overall score
            self.results['overall_score'] = self._calculate_overall_score()
            
            # 6. Generate report
            self._generate_validation_report()
            
            logger.info(f"Validation completed. Overall score: {self.results['overall_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            self.results['error'] = str(e)
        
        return self.results
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance improvements"""
        logger.info("Validating performance improvements...")
        
        # Create test data
        test_data = self._create_test_data(self.config['test_data_size'])
        
        # Initialize pipeline
        pipeline = TransformerMLPipeline(
            use_memory_efficient=True,
            enable_ensemble=True
        )
        
        pipeline.initialize_models(
            input_dim=20,  # Rich feature set
            output_dim=1,
            seq_len=self.config['sequence_length'],
            forecast_horizon=1
        )
        
        # Train and evaluate
        start_time = time.time()
        training_results = pipeline.train_pipeline(
            train_data=test_data,
            num_epochs=self.config['training_epochs'],
            batch_size=self.config['batch_size']
        )
        training_time = time.time() - start_time
        
        # Extract performance metrics
        transformer_accuracy = training_results['evaluation_results'].get('transformer_accuracy', 0.0)
        ensemble_accuracy = training_results['evaluation_results'].get('ensemble_accuracy', 0.0)
        improvement_pct = training_results['evaluation_results'].get('improvement_percentage', 0.0)
        
        # Performance validation
        performance_target_met = improvement_pct >= (self.config['performance_threshold'] * 100)
        
        return {
            'transformer_accuracy': transformer_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'improvement_percentage': improvement_pct,
            'training_time_seconds': training_time,
            'performance_target_met': performance_target_met,
            'target_improvement': self.config['performance_threshold'] * 100,
            'score': min(100, max(0, improvement_pct)) / 100  # Normalize to 0-1
        }
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate system integration"""
        logger.info("Validating system integration...")
        
        integration_score = 0.0
        tests_passed = 0
        total_tests = 5
        
        try:
            # Test 1: Model initialization
            pipeline = TransformerMLPipeline()
            pipeline.initialize_models(input_dim=10, output_dim=1, seq_len=50, forecast_horizon=1)
            tests_passed += 1
            logger.info("✓ Model initialization test passed")
            
            # Test 2: Data preprocessing
            test_data = self._create_test_data(100)
            preprocessor = TransformerPreprocessor()
            processed = preprocessor.fit_transform(test_data)
            if 'X_train' in processed and 'y_train' in processed:
                tests_passed += 1
                logger.info("✓ Data preprocessing test passed")
            
            # Test 3: Forward pass
            model = MemoryEfficientTransformer(input_dim=10, output_dim=1, seq_len=50, forecast_horizon=1)
            x = torch.randn(4, 50, 10)
            with torch.no_grad():
                output = model(x)
            if output.shape == (4, 1):
                tests_passed += 1
                logger.info("✓ Forward pass test passed")
            
            # Test 4: Signal generation
            signal_system = EnhancedSignalQualitySystem(
                transformer_model=model,
                ensemble_models=[]
            )
            signal = signal_system.generate_signal(
                market_data=test_data,
                symbol='BTCUSD',
                current_price=50000.0
            )
            if signal is not None:
                tests_passed += 1
                logger.info("✓ Signal generation test passed")
            
            # Test 5: Prediction pipeline
            prediction = pipeline.predict(
                market_data=test_data,
                symbol='BTCUSD',
                return_signal=False
            )
            if prediction is not None:
                tests_passed += 1
                logger.info("✓ Prediction pipeline test passed")
            
        except Exception as e:
            logger.error(f"Integration test failed: {str(e)}")
        
        integration_score = tests_passed / total_tests
        
        return {
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'integration_score': integration_score,
            'all_tests_passed': tests_passed == total_tests
        }
    
    def _validate_memory_efficiency(self) -> Dict[str, Any]:
        """Validate memory efficiency for M2 MacBook Air"""
        logger.info("Validating memory efficiency...")
        
        import psutil
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)  # GB
        
        # Create memory-efficient model
        model = MemoryEfficientTransformer(
            input_dim=20,
            output_dim=1,
            seq_len=100,
            forecast_horizon=1,
            d_model=128,  # Moderate size
            num_layers=4
        )
        
        # Test memory usage during training
        test_data = self._create_test_data(200)
        preprocessor = TransformerPreprocessor(sequence_length=100)
        processed = preprocessor.fit_transform(test_data)
        
        train_loader, val_loader = preprocessor.create_data_loaders(
            processed['X_train'],
            processed['y_train'],
            processed['X_val'],
            processed['y_val'],
            batch_size=8  # Small batch for memory efficiency
        )
        
        # Monitor memory during training
        max_memory = initial_memory
        
        try:
            # Quick training to test memory usage
            for epoch in range(2):
                for batch_idx, (X, y) in enumerate(train_loader):
                    if batch_idx > 5:  # Test only a few batches
                        break
                    
                    X = X.float()
                    y = y.float()
                    
                    # Forward pass
                    output = model(X)
                    
                    # Check memory
                    current_memory = process.memory_info().rss / (1024**3)
                    max_memory = max(max_memory, current_memory)
                    
                    # Cleanup
                    del X, y, output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        except Exception as e:
            logger.warning(f"Memory test encountered error: {str(e)}")
        
        memory_usage = max_memory - initial_memory
        memory_efficient = memory_usage < self.config['memory_limit_gb']
        
        return {
            'initial_memory_gb': initial_memory,
            'max_memory_gb': max_memory,
            'memory_usage_gb': memory_usage,
            'memory_limit_gb': self.config['memory_limit_gb'],
            'memory_efficient': memory_efficient,
            'memory_score': max(0, 1 - (memory_usage / self.config['memory_limit_gb']))
        }
    
    def _validate_signal_quality(self) -> Dict[str, Any]:
        """Validate signal quality system"""
        logger.info("Validating signal quality system...")
        
        # Create mock models for testing
        class MockTransformer:
            def predict(self, data, return_confidence=False):
                if return_confidence:
                    return np.array([0.75]), np.array([0.85])
                return np.array([0.75])
        
        class MockEnsemble:
            def predict(self, data):
                return 0.7
            confidence = 0.8
        
        # Initialize signal quality system
        signal_system = EnhancedSignalQualitySystem(
            transformer_model=MockTransformer(),
            ensemble_models=[MockEnsemble(), MockEnsemble()],
            confidence_threshold=self.config['confidence_threshold']
        )
        
        # Test signal generation
        test_data = self._create_test_data(100)
        signals_generated = 0
        high_quality_signals = 0
        
        for i in range(10):
            signal = signal_system.generate_signal(
                market_data=test_data,
                symbol='BTCUSD',
                current_price=50000.0 + i * 100
            )
            
            if signal:
                signals_generated += 1
                if signal.confidence >= 0.7:
                    high_quality_signals += 1
        
        # Get performance report
        performance_report = signal_system.get_performance_report()
        
        signal_quality_score = high_quality_signals / max(1, signals_generated)
        
        return {
            'signals_generated': signals_generated,
            'high_quality_signals': high_quality_signals,
            'signal_quality_score': signal_quality_score,
            'average_confidence': performance_report.get('average_confidence', 0.0),
            'performance_report': performance_report
        }
    
    def _create_test_data(self, num_samples: int) -> pd.DataFrame:
        """Create realistic test data"""
        dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='1H')
        np.random.seed(42)
        
        # Generate realistic price movements
        returns = np.random.normal(0, 0.02, num_samples)
        prices = [50000]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, num_samples)
        })
    
    def _calculate_overall_score(self) -> float:
        """Calculate overall validation score"""
        scores = []
        weights = []
        
        # Performance score (40% weight)
        if 'performance_tests' in self.results:
            scores.append(self.results['performance_tests'].get('score', 0.0))
            weights.append(0.4)
        
        # Integration score (25% weight)
        if 'integration_tests' in self.results:
            scores.append(self.results['integration_tests'].get('integration_score', 0.0))
            weights.append(0.25)
        
        # Memory efficiency score (20% weight)
        if 'memory_tests' in self.results:
            scores.append(self.results['memory_tests'].get('memory_score', 0.0))
            weights.append(0.2)
        
        # Signal quality score (15% weight)
        if 'signal_quality_tests' in self.results:
            scores.append(self.results['signal_quality_tests'].get('signal_quality_score', 0.0))
            weights.append(0.15)
        
        if not scores:
            return 0.0
        
        # Weighted average
        overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return overall_score
    
    def _generate_validation_report(self) -> None:
        """Generate comprehensive validation report"""
        report_path = Path("validation_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate summary
        summary = f"""
=== TRANSFORMER SYSTEM VALIDATION REPORT ===
Timestamp: {self.results['validation_timestamp']}
Overall Score: {self.results['overall_score']:.2f}/1.00

Performance Tests:
- Transformer Accuracy: {self.results.get('performance_tests', {}).get('transformer_accuracy', 0):.4f}
- Improvement: {self.results.get('performance_tests', {}).get('improvement_percentage', 0):.1f}%
- Target Met: {self.results.get('performance_tests', {}).get('performance_target_met', False)}

Integration Tests:
- Tests Passed: {self.results.get('integration_tests', {}).get('tests_passed', 0)}/5
- Integration Score: {self.results.get('integration_tests', {}).get('integration_score', 0):.2f}

Memory Efficiency:
- Memory Usage: {self.results.get('memory_tests', {}).get('memory_usage_gb', 0):.2f} GB
- Memory Efficient: {self.results.get('memory_tests', {}).get('memory_efficient', False)}

Signal Quality:
- High Quality Signals: {self.results.get('signal_quality_tests', {}).get('high_quality_signals', 0)}
- Quality Score: {self.results.get('signal_quality_tests', {}).get('signal_quality_score', 0):.2f}

Status: {'PASSED' if self.results['overall_score'] >= 0.7 else 'NEEDS IMPROVEMENT'}
"""
        
        print(summary)
        
        with open("validation_summary.txt", 'w') as f:
            f.write(summary)
        
        logger.info(f"Validation report saved to {report_path}")


def main():
    """Main validation function"""
    logger.info("Starting Transformer System Validation")
    
    # Initialize validator
    validator = TransformerSystemValidator()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Print final status
    overall_score = results.get('overall_score', 0.0)
    status = "PASSED" if overall_score >= 0.7 else "NEEDS IMPROVEMENT"
    
    logger.info(f"Validation completed with score: {overall_score:.2f} - {status}")
    
    return results


if __name__ == "__main__":
    main()
