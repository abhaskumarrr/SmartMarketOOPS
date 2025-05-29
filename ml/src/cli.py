"""
Command Line Interface for SMOOPs ML System

This module provides a command-line interface for interacting with the ML system.
"""

import os
import sys
import argparse
import logging
import json
from .backtesting.engine import BacktestEngine
from .backtesting.strategies import SmcBasedStrategy
from .backtesting.utils import plot_backtest_results
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_command(args):
    """
    Train a new model based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    from .training.train_model import train_model
    
    logger.info("Starting model training")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Symbol: {args.symbol}")
    
    # Parse additional training args from JSON if provided
    training_args = {}
    if args.training_args:
        try:
            training_args = json.loads(args.training_args)
            logger.info(f"Additional training args: {training_args}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON for training args: {args.training_args}")
    
    # Train the model
    model_info = train_model(
        symbol=args.symbol,
        model_type=args.model_type,
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        **training_args
    )
    
    logger.info(f"Model training completed. Model saved as {model_info.get('version')}")
    logger.info(f"Metrics: {model_info.get('metrics')}")
    
    return model_info

def predict_command(args):
    """
    Make predictions using a trained model.
    
    Args:
        args: Command line arguments
    """
    from .models.model_registry import ModelRegistry
    
    logger.info("Making predictions")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Model version: {args.version or 'latest'}")
    
    # Load the model and preprocessor
    registry = ModelRegistry()
    model, _, preprocessor = registry.load_model(
        symbol=args.symbol,
        version=args.version,
        return_metadata=True,
        return_preprocessor=True
    )
    
    # Load data
    if args.data_file:
        df = pd.read_csv(args.data_file)
    else:
        # Use API to get latest data
        from .data.market_data import get_latest_data
        df = get_latest_data(args.symbol, args.time_frame, args.limit)
    
    # Feature engineering to match training pipeline
    from .data.data_loader import MarketDataLoader
    loader = MarketDataLoader(timeframe='1h', symbols=[args.symbol])
    df_proc = loader.preprocess_for_smc(df)
    X_df, _ = loader.create_features(df_proc)
    # Use only the feature columns used in training
    df_numeric = X_df
    if preprocessor is not None:
        # If the preprocessor is a dict (feature_scaler, target_scaler), use feature_scaler
        if isinstance(preprocessor, dict) and 'feature_scaler' in preprocessor:
            X = preprocessor['feature_scaler'].transform(df_numeric.values)
        else:
            X = preprocessor.transform(df_numeric.values)
    else:
        logger.warning("No preprocessor found for this model version. Using raw input data.")
        X = df_numeric.values
    
    # Reshape X for sequence models if needed (assume sequence_length from model config or args)
    sequence_length = getattr(model, 'seq_len', None) or getattr(args, 'sequence_length', 60)
    if len(X.shape) == 2 and X.shape[0] >= sequence_length:
        X_seq = []
        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X[i:i+sequence_length])
        X = np.array(X_seq)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Save or return predictions
    if args.output_file:
        pd.DataFrame(predictions).to_csv(args.output_file, index=False)
        logger.info(f"Predictions saved to {args.output_file}")
    else:
        print(predictions)
    
    return predictions

def serve_command(args):
    """
    Start the model serving API.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting model serving API")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    
    # Import here to avoid circular imports
    import uvicorn
    
    # Start the server (assuming server.py handles the actual app creation)
    uvicorn.run(
        "ml.backend.src.scripts.server:app", 
        host=args.host, 
        port=args.port,
        reload=True
    )

def evaluate_command(args):
    """
    Evaluate a trained model.
    
    Args:
        args: Command line arguments
    """
    from .models.model_registry import ModelRegistry
    from .data.data_loader import load_data
    from .training.evaluation import evaluate_model
    
    logger.info("Starting model evaluation")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Model version: {args.version or 'latest'}")
    
    # Load the model
    registry = ModelRegistry()
    model = registry.load_model(
        symbol=args.symbol,
        version=args.version
    )
    
    # Load test data
    _, _, test_dataloader = load_data(
        symbol=args.symbol,
        data_path=args.data_path,
        train_ratio=0,
        val_ratio=0,
        test_ratio=1.0,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_dataloader)
    
    # Save or print metrics
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {args.output_file}")
    else:
        print(json.dumps(metrics, indent=2))
    
    return metrics

def backtest_command(args):
    """
    Run a backtest using the BacktestEngine and selected strategy.
    """
    logger.info("Starting backtest...")
    # Load data
    data = pd.read_csv(args.data_file)
    # Select strategy (expandable)
    if args.strategy == "smc":
        strategy = SmcBasedStrategy(name="SMC Strategy")
    elif args.strategy == "ml_model":
        # Feature engineering to match training pipeline
        from ml.src.data.data_loader import MarketDataLoader
        loader = MarketDataLoader(timeframe='1h', symbols=[args.symbol])
        df_proc = loader.preprocess_for_smc(data)
        X_df, _ = loader.create_features(df_proc)
        feature_cols = list(X_df.columns)
        input_dim = len(feature_cols)
        logger.info(f"MLModelStrategy feature columns: {feature_cols}")
        output_dim = args.forecast_horizon if hasattr(args, 'forecast_horizon') and args.forecast_horizon else 1
        seq_len = args.sequence_length if hasattr(args, 'sequence_length') and args.sequence_length else 60
        forecast_horizon = output_dim
        # Dynamically import model class
        from ml.src.models import lstm_model, gru_model, transformer_model, cnn_lstm_model
        model_classes = {
            'lstm': lstm_model.LSTMModel,
            'gru': gru_model.GRUModel,
            'transformer': transformer_model.TransformerModel,
            'cnn_lstm': cnn_lstm_model.CNNLSTMModel
        }
        model_class = model_classes[args.model_type]
        model_kwargs = dict(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon
        )
        from .backtesting.strategies import MLModelStrategy
        strategy = MLModelStrategy(
            model_class=model_class,
            model_kwargs=model_kwargs,
            model_checkpoint=args.model_checkpoint,
            preprocessor_path=args.preprocessor,
            device=args.device if hasattr(args, 'device') else 'cpu',
            threshold=args.threshold,
            name=f"MLModelStrategy-{args.model_type}"
        )
        # Re-attach timestamp column for backtest engine
        if 'timestamp' in data.columns:
            X_df = X_df.copy()
            X_df.insert(0, 'timestamp', data['timestamp'].iloc[-len(X_df):].values)
        data = X_df
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        sys.exit(1)
    # Initialize engine
    engine = BacktestEngine(
        strategy,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage_factor=args.slippage,
        logging_level=args.logging_level
    )
    # Run backtest
    results = engine.run(
        data,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    # Print summary
    def to_python_type(obj):
        if isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_type(v) for v in obj]
        elif hasattr(obj, 'item') and callable(obj.item):
            return obj.item()
        else:
            return obj
    print("Backtest Results:")
    print(json.dumps(to_python_type(results['metrics']), indent=2))
    # Save results/plots if output_dir
    if args.output_dir:
        engine.save_results(os.path.join(args.output_dir, f"backtest_{args.symbol}.json"))
        engine.plot_results(output_dir=args.output_dir)
    else:
        plot_backtest_results(results)
    logger.info("Backtest complete.")

def main():
    """Main entry point for the command line interface"""
    parser = argparse.ArgumentParser(description="SMOOPs ML System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., BTC-USDT)")
    train_parser.add_argument("--model-type", type=str, required=True, help="Type of model to train")
    train_parser.add_argument("--data-path", type=str, help="Path to input data file")
    train_parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion of data for training")
    train_parser.add_argument("--val-ratio", type=float, default=0.15, help="Proportion of data for validation")
    train_parser.add_argument("--test-ratio", type=float, default=0.15, help="Proportion of data for testing")
    train_parser.add_argument("--sequence-length", type=int, default=60, help="Input sequence length")
    train_parser.add_argument("--forecast-horizon", type=int, default=1, help="Number of steps to predict")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    train_parser.add_argument("--num-epochs", type=int, default=100, help="Maximum number of epochs")
    train_parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    train_parser.add_argument("--training-args", type=str, help="Additional training args as JSON")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a trained model")
    predict_parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., BTC-USDT)")
    predict_parser.add_argument("--version", type=str, help="Model version (default: latest)")
    predict_parser.add_argument("--data-file", type=str, help="Input data file for prediction")
    predict_parser.add_argument("--time-frame", type=str, default="1h", help="Timeframe for fetching data")
    predict_parser.add_argument("--limit", type=int, default=100, help="Number of candles to fetch")
    predict_parser.add_argument("--output-file", type=str, help="Output file for predictions")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the model serving API")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    serve_parser.add_argument("--port", type=int, default=3002, help="API port")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--symbol", type=str, required=True, help="Trading symbol (e.g., BTC-USDT)")
    evaluate_parser.add_argument("--version", type=str, help="Model version (default: latest)")
    evaluate_parser.add_argument("--data-path", type=str, help="Path to test data")
    evaluate_parser.add_argument("--sequence-length", type=int, default=60, help="Input sequence length")
    evaluate_parser.add_argument("--forecast-horizon", type=int, default=1, help="Number of steps to predict")
    evaluate_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    evaluate_parser.add_argument("--output-file", type=str, help="Output file for metrics")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest with historical data")
    backtest_parser.add_argument("--data-file", type=str, required=True, help="Path to CSV file with OHLCV data")
    backtest_parser.add_argument("--strategy", type=str, default="smc", help="Strategy to use (default: smc)")
    backtest_parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    backtest_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    backtest_parser.add_argument("--fee-rate", type=float, default=0.001, help="Trading fee rate (e.g., 0.001 for 0.1%)")
    backtest_parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage factor (e.g., 0.0005 for 0.05%)")
    backtest_parser.add_argument("--output-dir", type=str, help="Directory to save results and plots")
    backtest_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--logging-level", type=str, default="INFO", help="Logging level (default: INFO)")
    backtest_parser.add_argument("--model-type", type=str, help="ML model type (lstm, gru, transformer, cnn_lstm, etc.) for MLModelStrategy")
    backtest_parser.add_argument("--model-checkpoint", type=str, help="Path to PyTorch model checkpoint (.pt/.pth) for MLModelStrategy")
    backtest_parser.add_argument("--preprocessor", type=str, help="Path to preprocessor (joblib/pickle) for MLModelStrategy")
    backtest_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification/regression in MLModelStrategy")
    backtest_parser.add_argument("--device", type=str, help="Device to use for MLModelStrategy")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "serve":
        serve_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "backtest":
        backtest_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 