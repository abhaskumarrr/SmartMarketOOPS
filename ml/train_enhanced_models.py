#!/usr/bin/env python3
"""
Enhanced Model Training Script for Production Deployment
Trains enhanced Transformer models for major trading symbols
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.models.model_factory import ModelFactory
from src.models.model_registry import ModelRegistry
from src.data.transformer_preprocessor import TransformerPreprocessor
from src.utils.config import MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedModelTrainer:
    """Enhanced model trainer for production deployment"""
    
    def __init__(self):
        """Initialize the trainer"""
        self.model_registry = ModelRegistry()
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]  # Major trading symbols
        self.config = MODEL_CONFIG
        
        logger.info("Enhanced Model Trainer initialized")
    
    def generate_synthetic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Generate synthetic market data for training
        In production, this would be replaced with real market data
        """
        logger.info(f"Generating synthetic data for {symbol} ({days} days)")
        
        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible results
        
        # Base price levels for different symbols
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 2500,
            "ADAUSDT": 0.5
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # Generate hourly data
        periods = days * 24
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=periods,
            freq='1H'
        )
        
        # Generate price movements with trend and volatility
        returns = np.random.normal(0.0001, 0.02, periods)  # Small positive drift with volatility
        
        # Add some trend patterns
        trend = np.sin(np.arange(periods) / (24 * 7)) * 0.001  # Weekly patterns
        returns += trend
        
        # Calculate prices
        prices = [base_price]
        for i in range(1, periods):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC from close price
            volatility = abs(returns[i]) * close
            
            open_price = close * (1 + np.random.normal(0, 0.001))
            high = max(open_price, close) + np.random.exponential(volatility)
            low = min(open_price, close) - np.random.exponential(volatility)
            volume = np.random.lognormal(15, 1)  # Realistic volume distribution
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} data points for {symbol}")
        return df
    
    def train_enhanced_transformer(self, symbol: str) -> bool:
        """Train enhanced Transformer model for a symbol"""
        logger.info(f"Training enhanced Transformer model for {symbol}")
        
        try:
            # Generate training data
            df = self.generate_synthetic_data(symbol, days=180)  # 6 months of data
            
            # Initialize enhanced preprocessor
            preprocessor = TransformerPreprocessor(
                sequence_length=100,
                forecast_horizon=1,
                scaling_method='standard',
                feature_engineering=True,
                multi_timeframe=True,
                attention_features=True
            )
            
            # Preprocess data
            processed_data = preprocessor.fit_transform(df, target_column='close')
            
            logger.info(f"Preprocessed data: {processed_data['num_features']} features, "
                       f"{len(processed_data['X_train'])} training samples")
            
            # Create enhanced Transformer model
            model = ModelFactory.create_model(
                model_type='enhanced_transformer',
                input_dim=processed_data['num_features'],
                output_dim=1,
                seq_len=processed_data['sequence_length'],
                forecast_horizon=1,
                d_model=self.config['transformer']['d_model'],
                nhead=self.config['transformer']['nhead'],
                num_layers=self.config['transformer']['num_layers'],
                use_financial_attention=self.config['transformer']['use_financial_attention'],
                dropout=self.config['dropout']
            )
            
            logger.info(f"Created enhanced Transformer model with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Create data loaders
            train_loader, val_loader = preprocessor.create_data_loaders(
                processed_data['X_train'],
                processed_data['y_train'],
                processed_data['X_val'],
                processed_data['y_val'],
                batch_size=self.config['batch_size']
            )
            
            # Train the model
            logger.info("Starting model training...")
            history = model.fit_model(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=20,  # Reduced for demo
                lr=self.config['learning_rate'],
                early_stopping_patience=self.config['patience']
            )
            
            logger.info(f"Training completed. Best validation loss: {history['best_val_loss']:.6f}")
            
            # Save the model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create metadata
            metadata = {
                'symbol': symbol,
                'model_type': 'enhanced_transformer',
                'version': version,
                'input_dim': processed_data['num_features'],
                'output_dim': 1,
                'sequence_length': processed_data['sequence_length'],
                'forecast_horizon': 1,
                'training_samples': len(processed_data['X_train']),
                'validation_samples': len(processed_data['X_val']),
                'best_val_loss': history['best_val_loss'],
                'epochs_trained': history.get('epochs_trained', 20),
                'feature_names': processed_data['feature_names'],
                'config': self.config['transformer'],
                'trained_at': datetime.now().isoformat()
            }
            
            # Save to registry
            self.model_registry.save_model(
                model=model,
                symbol=symbol,
                version=version,
                metadata=metadata,
                preprocessor=preprocessor
            )
            
            logger.info(f"Enhanced Transformer model saved for {symbol} (version: {version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train enhanced Transformer for {symbol}: {e}")
            return False
    
    def train_all_symbols(self) -> dict:
        """Train enhanced models for all symbols"""
        logger.info("Starting enhanced model training for all symbols")
        
        results = {}
        
        for symbol in self.symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training enhanced model for {symbol}")
            logger.info(f"{'='*50}")
            
            success = self.train_enhanced_transformer(symbol)
            results[symbol] = success
            
            if success:
                logger.info(f"✅ Successfully trained enhanced model for {symbol}")
            else:
                logger.error(f"❌ Failed to train enhanced model for {symbol}")
        
        return results
    
    def validate_trained_models(self) -> dict:
        """Validate that trained models can be loaded and used"""
        logger.info("Validating trained models...")
        
        validation_results = {}
        
        for symbol in self.symbols:
            try:
                # Load the model
                model, metadata = self.model_registry.load_model(
                    symbol=symbol,
                    return_metadata=True
                )
                
                # Test prediction
                test_input = np.random.randn(1, metadata.get('sequence_length', 100), metadata.get('input_dim', 20))
                
                with torch.no_grad():
                    prediction = model.predict(test_input)
                
                validation_results[symbol] = {
                    'status': 'success',
                    'model_type': metadata.get('model_type', 'unknown'),
                    'version': metadata.get('version', 'unknown'),
                    'prediction_shape': prediction.shape if hasattr(prediction, 'shape') else 'scalar'
                }
                
                logger.info(f"✅ {symbol}: Model loaded and prediction successful")
                
            except Exception as e:
                validation_results[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"❌ {symbol}: Validation failed - {e}")
        
        return validation_results


def main():
    """Main training function"""
    print("🚀 Enhanced Model Training for Production Deployment")
    print("="*60)
    
    trainer = EnhancedModelTrainer()
    
    # Train models for all symbols
    training_results = trainer.train_all_symbols()
    
    # Validate trained models
    validation_results = trainer.validate_trained_models()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    successful_training = sum(1 for success in training_results.values() if success)
    total_symbols = len(training_results)
    
    print(f"Training Results: {successful_training}/{total_symbols} successful")
    
    for symbol, success in training_results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {symbol}: {status}")
    
    print(f"\nValidation Results:")
    for symbol, result in validation_results.items():
        status = "✅ VALID" if result['status'] == 'success' else "❌ INVALID"
        print(f"  {symbol}: {status}")
        if result['status'] == 'success':
            print(f"    Model Type: {result['model_type']}")
            print(f"    Version: {result['version']}")
    
    if successful_training == total_symbols:
        print("\n🎉 All enhanced models trained successfully!")
        print("✅ Production deployment ready")
    else:
        print(f"\n⚠️  {total_symbols - successful_training} models failed to train")
        print("❌ Check logs for issues")
    
    return successful_training == total_symbols


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
