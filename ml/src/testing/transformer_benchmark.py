"""
Transformer Model Benchmarking and Performance Testing
Implements Subtask 24.4: Performance Optimization and Benchmarking
Validates 20-30% performance improvement target
"""

import time
import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

from ml.src.models.transformer_model import EnhancedTransformerModel, TransformerModel
from ml.src.models.lstm_model import LSTMModel
from ml.src.models.cnn_lstm_model import CNNLSTMModel
from ml.src.data.transformer_preprocessor import TransformerPreprocessor
from ml.src.data.market_data_loader import MarketDataLoader
from ml.src.utils.config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class TransformerBenchmark:
    """
    Comprehensive benchmarking suite for Transformer models
    Compares performance against LSTM/CNN baselines
    """
    
    def __init__(self, symbol: str = "BTCUSDT", days_back: int = 365):
        """
        Initialize benchmark with market data
        
        Args:
            symbol: Trading symbol to test on
            days_back: Number of days of historical data
        """
        self.symbol = symbol
        self.days_back = days_back
        self.results = {}
        
        # Load and preprocess data
        self.data_loader = MarketDataLoader()
        self.preprocessor = TransformerPreprocessor(
            sequence_length=100,
            forecast_horizon=1,
            scaling_method='standard',
            feature_engineering=True,
            multi_timeframe=True,
            attention_features=True
        )
        
        logger.info(f"TransformerBenchmark initialized for {symbol}")
    
    def load_data(self) -> Dict[str, Any]:
        """Load and preprocess market data"""
        logger.info(f"Loading {self.days_back} days of {self.symbol} data")
        
        # Load raw data
        df = self.data_loader.get_data(
            symbol=self.symbol,
            interval="1h",
            days_back=self.days_back,
            use_cache=True
        )
        
        # Preprocess for Transformer
        processed_data = self.preprocessor.fit_transform(df, target_column='close')
        
        # Create data loaders
        train_loader, val_loader = self.preprocessor.create_data_loaders(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            batch_size=32
        )
        
        self.processed_data = processed_data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        logger.info(f"Data loaded: {processed_data['num_features']} features, "
                   f"train: {len(processed_data['X_train'])}, val: {len(processed_data['X_val'])}")
        
        return processed_data
    
    def create_models(self) -> Dict[str, torch.nn.Module]:
        """Create all models for comparison"""
        input_dim = self.processed_data['num_features']
        seq_len = self.processed_data['sequence_length']
        
        models = {
            'enhanced_transformer': EnhancedTransformerModel(
                input_dim=input_dim,
                output_dim=1,
                seq_len=seq_len,
                forecast_horizon=1,
                d_model=256,
                nhead=8,
                num_layers=6,
                dropout=0.1,
                use_financial_attention=True
            ),
            'standard_transformer': TransformerModel(
                input_dim=input_dim,
                output_dim=1,
                seq_len=seq_len,
                forecast_horizon=1,
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.2
            ),
            'lstm': LSTMModel(
                input_dim=input_dim,
                output_dim=1,
                seq_len=seq_len,
                forecast_horizon=1,
                hidden_dim=128,
                num_layers=2,
                dropout=0.2
            ),
            'cnn_lstm': CNNLSTMModel(
                input_size=input_dim,
                cnn_channels=64,
                lstm_hidden=128,
                lstm_layers=2,
                dropout=0.3,
                num_classes=1
            )
        }
        
        logger.info(f"Created {len(models)} models for benchmarking")
        return models
    
    def benchmark_training_speed(self, models: Dict[str, torch.nn.Module]) -> Dict[str, float]:
        """Benchmark training speed for each model"""
        logger.info("Benchmarking training speed...")
        
        training_times = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            start_time = time.time()
            
            try:
                # Train for a few epochs to measure speed
                history = model.fit_model(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=5,  # Short training for speed test
                    lr=0.001,
                    early_stopping_patience=10
                )
                
                training_time = time.time() - start_time
                training_times[name] = training_time
                
                logger.info(f"{name} training time: {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                training_times[name] = float('inf')
        
        return training_times
    
    def benchmark_inference_speed(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        """Benchmark inference speed and latency"""
        logger.info("Benchmarking inference speed...")
        
        # Prepare test data
        test_data = self.processed_data['X_val'][:100]  # Use first 100 samples
        batch_sizes = [1, 16, 32, 64]
        
        inference_results = {}
        
        for name, model in models.items():
            model.eval()
            inference_results[name] = {}
            
            for batch_size in batch_sizes:
                times = []
                
                # Warm up
                for _ in range(5):
                    with torch.no_grad():
                        batch = torch.FloatTensor(test_data[:batch_size])
                        _ = model(batch)
                
                # Measure inference time
                for i in range(0, len(test_data), batch_size):
                    batch = torch.FloatTensor(test_data[i:i+batch_size])
                    
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(batch)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                avg_latency = avg_time / batch_size * 1000  # ms per sample
                
                inference_results[name][f'batch_{batch_size}'] = {
                    'avg_time': avg_time,
                    'latency_ms': avg_latency,
                    'throughput': batch_size / avg_time
                }
                
                logger.info(f"{name} (batch {batch_size}): {avg_latency:.2f}ms/sample")
        
        return inference_results
    
    def benchmark_accuracy(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        """Benchmark prediction accuracy"""
        logger.info("Benchmarking prediction accuracy...")
        
        accuracy_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name} for accuracy benchmark...")
            
            try:
                # Train model properly
                history = model.fit_model(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    num_epochs=50,
                    lr=0.001,
                    early_stopping_patience=10
                )
                
                # Evaluate on validation set
                metrics = model.evaluate(
                    self.processed_data['X_val'],
                    self.processed_data['y_val']
                )
                
                accuracy_results[name] = {
                    'val_loss': history.get('best_val_loss', float('inf')),
                    **metrics
                }
                
                logger.info(f"{name} accuracy: {metrics.get('accuracy', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
                accuracy_results[name] = {'error': str(e)}
        
        return accuracy_results
    
    def benchmark_memory_usage(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        """Benchmark memory usage"""
        logger.info("Benchmarking memory usage...")
        
        memory_results = {}
        
        for name, model in models.items():
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage (rough approximation)
            param_memory = total_params * 4 / (1024**2)  # MB (assuming float32)
            
            memory_results[name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'param_memory_mb': param_memory
            }
            
            logger.info(f"{name}: {total_params:,} params, {param_memory:.1f}MB")
        
        return memory_results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive Transformer benchmark")
        
        # Load data
        self.load_data()
        
        # Create models
        models = self.create_models()
        
        # Run all benchmarks
        results = {
            'data_info': {
                'symbol': self.symbol,
                'days_back': self.days_back,
                'num_features': self.processed_data['num_features'],
                'train_samples': len(self.processed_data['X_train']),
                'val_samples': len(self.processed_data['X_val'])
            },
            'training_speed': self.benchmark_training_speed(models),
            'inference_speed': self.benchmark_inference_speed(models),
            'accuracy': self.benchmark_accuracy(models),
            'memory_usage': self.benchmark_memory_usage(models)
        }
        
        self.results = results
        
        # Calculate improvement metrics
        self._calculate_improvements()
        
        return results
    
    def _calculate_improvements(self):
        """Calculate improvement percentages vs baseline models"""
        if 'accuracy' not in self.results:
            return
        
        # Use LSTM as baseline
        baseline_accuracy = self.results['accuracy'].get('lstm', {}).get('accuracy', 0)
        enhanced_accuracy = self.results['accuracy'].get('enhanced_transformer', {}).get('accuracy', 0)
        
        if baseline_accuracy > 0 and enhanced_accuracy > 0:
            improvement = ((enhanced_accuracy - baseline_accuracy) / baseline_accuracy) * 100
            self.results['improvement_vs_lstm'] = improvement
            
            logger.info(f"Enhanced Transformer improvement vs LSTM: {improvement:.1f}%")
            
            # Check if we meet the 20-30% target
            if improvement >= 20:
                logger.info("✅ Target 20-30% improvement ACHIEVED!")
            else:
                logger.warning(f"❌ Target 20-30% improvement NOT met (got {improvement:.1f}%)")
    
    def save_results(self, filepath: str):
        """Save benchmark results to file"""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            logger.error("No results to display. Run benchmark first.")
            return
        
        print("\n" + "="*60)
        print("TRANSFORMER BENCHMARK SUMMARY")
        print("="*60)
        
        # Accuracy comparison
        if 'accuracy' in self.results:
            print("\nACCURACY COMPARISON:")
            for model, metrics in self.results['accuracy'].items():
                if 'accuracy' in metrics:
                    print(f"  {model:20}: {metrics['accuracy']:.4f}")
        
        # Speed comparison
        if 'inference_speed' in self.results:
            print("\nINFERENCE LATENCY (ms/sample, batch=1):")
            for model, speeds in self.results['inference_speed'].items():
                if 'batch_1' in speeds:
                    latency = speeds['batch_1']['latency_ms']
                    print(f"  {model:20}: {latency:.2f}ms")
        
        # Memory usage
        if 'memory_usage' in self.results:
            print("\nMEMORY USAGE:")
            for model, memory in self.results['memory_usage'].items():
                params = memory['total_params']
                memory_mb = memory['param_memory_mb']
                print(f"  {model:20}: {params:,} params, {memory_mb:.1f}MB")
        
        # Improvement summary
        if 'improvement_vs_lstm' in self.results:
            improvement = self.results['improvement_vs_lstm']
            print(f"\nIMPROVEMENT vs LSTM: {improvement:.1f}%")
            if improvement >= 20:
                print("✅ TARGET ACHIEVED (20-30% improvement)")
            else:
                print("❌ TARGET NOT MET")
        
        print("="*60)


def main():
    """Run the benchmark"""
    logging.basicConfig(level=logging.INFO)
    
    # Run benchmark
    benchmark = TransformerBenchmark(symbol="BTCUSDT", days_back=180)
    results = benchmark.run_comprehensive_benchmark()
    
    # Print summary
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results("transformer_benchmark_results.json")


if __name__ == "__main__":
    main()
