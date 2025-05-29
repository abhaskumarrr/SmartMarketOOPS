import torch
import torch.nn as nn
import torch.optim as optim
from ml.backend.src.data.loader import get_dataloader
# from ml.backend.src.models.cnn_lstm import CNNLSTMModel # Corrected import below
from ml.src.models.cnn_lstm_model import CNNLSTMModel # Import CNNLSTMModel from its new location
from ml.backend.src.training.trainer import Trainer
import os
from datetime import datetime # Import datetime
# import matplotlib.pyplot as plt # Removed matplotlib import
# import numpy as np # Removed numpy import for metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # Import classification metrics
import logging # Import logging

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Config
csv_file = 'sample_data/BTCUSD_15m.csv'
schema_file = 'ml/backend/src/data/schema.yaml'
batch_size = 16
seq_len = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define class weights for handling imbalance (calculated based on sample data distribution)
# Weights are inversely proportional to class frequencies: [count_0, count_1, count_2] = [72735, 262980, 70956] -> Total = 406671
# Weights = [Total/count_0, Total/count_1, Total/count_2]
class_weights = torch.tensor([406671/72735, 406671/262980, 406671/70956], dtype=torch.float32).to(device)

if __name__ == "__main__":
    import argparse
    from ml.backend.src.training.hyperparameter_tuning import run_optuna
    from ml.backend.src.data.loader import TradingDataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperopt', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--csv_file', type=str, default=csv_file)
    parser.add_argument('--schema_file', type=str, default=schema_file)
    parser.add_argument('--seq_len', type=int, default=seq_len)
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds for Optuna batch run')
    args = parser.parse_args()

    # Define model and normalization parameter save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join('models', 'registry', 'BTCUSD', f'v_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, 'best.pt')
    norm_params_file = os.path.join(model_dir, 'norm_params.npy') # Update norm_params_file path

    if args.hyperopt:
        try:
            # Load full dataset
            dataset = TradingDataset(args.csv_file, args.schema_file, seq_len=args.seq_len, norm_params_file=norm_params_file, save_norm_params=True)
            # Time-based split (80% train, 20% val)
            n = len(dataset)
            split = int(n * 0.8)
            train_dataset = torch.utils.data.Subset(dataset, range(0, split))
            val_dataset = torch.utils.data.Subset(dataset, range(split, n))
            # from ml.backend.src.models.cnn_lstm import CNNLSTMModel # Corrected import below
            from ml.src.models.cnn_lstm_model import CNNLSTMModel # Import CNNLSTMModel from its new location
            # Add Optuna MedianPruner for early stopping
            import optuna
            pruner = optuna.pruners.MedianPruner()
            # Pass num_classes to run_optuna and model initialization
            best_trial = run_optuna(CNNLSTMModel, train_dataset, val_dataset, device, n_trials=args.n_trials, timeout=args.timeout, pruner=pruner, class_weights=class_weights) # Pass class_weights to run_optuna
            print('Optuna study is robust and resumable. If interrupted, just rerun to continue from last trial.')
            best_params = best_trial.params
            # Retrain best model on full train set
            # Pass num_classes to model initialization
            model = CNNLSTMModel(input_size=train_dataset[0][0].shape[-1], cnn_channels=best_params['cnn_channels'], lstm_hidden=best_params['lstm_hidden'], lstm_layers=best_params['lstm_layers'], dropout=best_params['dropout'], num_classes=3)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights) # Use weighted loss
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
            trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
            trainer.train(num_epochs=30)
            
            # Define model configuration to save
            model_config_to_save = {
                "model_type": "cnn_lstm",
                "input_size": train_dataset[0][0].shape[-1], # Get actual input size from data
                "num_classes": 3, # Use actual num_classes
                "seq_len": args.seq_len, # Use actual seq_len
                "hidden_dim": best_params['lstm_hidden'], # Use actual lstm_hidden
                "num_layers": best_params['lstm_layers'], # Use actual lstm_layers
                "dropout": best_params['dropout'], # Use actual dropout
                "cnn_channels": best_params['cnn_channels'], # Use actual cnn_channels
                "lstm_hidden": best_params['lstm_hidden'], # Redundant but for clarity
                "lstm_layers": best_params['lstm_layers'] # Redundant but for clarity
            }

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'hyperparams': best_params,
                'config': model_config_to_save # Include the config here
            }
            # os.makedirs('models', exist_ok=True) # Original models dir creation
            torch.save(checkpoint, model_save_path) # Save to dynamic path
            print(f'Best model retrained and saved to {model_save_path}')
        except Exception as e:
            print(f"[ERROR] Exception during hyperopt and retrain: {e}")
            raise
        
        # Add ModelRegistry.save_model call here
        try:
            from ml.src.models.model_registry import ModelRegistry # Import ModelRegistry
            registry = ModelRegistry()
            # Need to get metrics and scaler in hyperopt block too
            # For simplicity, let's just use dummy metrics for now, or ideally, evaluate on the validation set again.
            # Let's evaluate on the validation set again to get proper metrics.
            model.eval()
            preds = []
            actuals = []
            # Assuming val_dataset is available from the hyperopt scope
            val_loader_for_save = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            with torch.no_grad():
                for window, target in val_loader_for_save:
                    window = window.to(device)
                    outputs = model(window) # Get raw outputs (logits)
                    _, predicted = torch.max(outputs.data, 1) # Get predicted class index
                    preds.extend(predicted.cpu().numpy())
                    actuals.extend(target.cpu().numpy())
            
            # Classification metrics
            accuracy = accuracy_score(actuals, preds)
            precision = precision_score(actuals, preds, average='weighted', zero_division=0)
            recall = recall_score(actuals, preds, average='weighted', zero_division=0)
            f1 = f1_score(actuals, preds, average='weighted', zero_division=0)

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

            # Assuming dataset object from hyperopt scope has scaler attribute
            scaler = dataset.scaler if hasattr(dataset, 'scaler') else None # Use dataset from hyperopt scope

            registry.save_model(
                model=model,
                symbol="BTC/USDT", # Ensure symbol matches the directory name expected by model_service
                metadata={
                    "model_type": "cnn_lstm", # Ensure consistent naming
                    "input_features": dataset.feature_cols if hasattr(dataset, 'feature_cols') else "Unknown",
                    "seq_len": seq_len, # Use seq_len from args
                    "architecture": "CNNLSTM",
                    "num_classes": 3,
                    "training_mode": "hyperopt"
                },
                preprocessor=scaler, # Save the scaler
                metrics=metrics
            )
            print(f'Best model from hyperopt saved via ModelRegistry to {model_dir}')

        except Exception as e:
            print(f"[ERROR] Exception during ModelRegistry save in hyperopt: {e}")
            # Do not re-raise here, allow the script to finish even if saving fails

    else:
        try:
            # Prepare DataLoader
            # Pass save_norm_params=True and norm_params_file to get_dataloader
            train_loader = get_dataloader(csv_file, schema_file, batch_size=batch_size, shuffle=True, seq_len=seq_len, norm_params_file=norm_params_file, save_norm_params=True, num_workers=0)
            val_loader = get_dataloader(csv_file, schema_file, batch_size=batch_size, shuffle=False, seq_len=seq_len, norm_params_file=norm_params_file, num_workers=0)
            # Model params (input_size = number of features excluding timestamp)
            sample = next(iter(train_loader))
            window, target = sample
            print('Sample window shape:', window.shape)  # (batch, seq_len, input_size)
            print('Sample target shape:', target.shape)  # (batch)
            input_size = window.shape[2]
            # Pass num_classes to model initialization
            model = CNNLSTMModel(input_size=input_size, num_classes=3)
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss(weight=class_weights) # Use weighted loss
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            # Instantiate dataset directly to print feature details (Optional, for verification)
            # temp_dataset = TradingDataset(csv_file, schema_file, seq_len=seq_len)
            # print(f"TradingDataset Feature Columns: {temp_dataset.feature_cols}")
            # print(f"TradingDataset Min values: {temp_params_file}")
            print('Starting training...')
            trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, log_dir='runs/cnnlstm')
            trainer.train(num_epochs=5)
            # Save model (no best_params in this mode)
            # os.makedirs('models', exist_ok=True) # Original models dir creation
            # Save num_classes with the model state dict
            # torch.save({'model_state_dict': model.state_dict(), 'model_config': {'model_type': 'cnnlstm', 'num_classes': 3}}, model_save_path) # Save to dynamic path
            print(f'Training complete. Model saved to {model_save_path}')
            # --- Evaluation ---
            model.eval()
            # Define model configuration to save in standard training block
            model_config_to_save = {
                "model_type": "cnn_lstm", # Ensure consistent naming
                "input_size": input_size, # Use actual input size
                "num_classes": 3, # Use actual num_classes
                "seq_len": seq_len, # Use actual seq_len
                # Use default values for other params as hyperopt wasn't run
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.3,
                "cnn_channels": 64,
                "lstm_hidden": 128,
                "lstm_layers": 2
            }

            # Save model using torch.save with config
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': model_config_to_save
            }
            torch.save(checkpoint, model_save_path)

            preds = []
            actuals = []
            with torch.no_grad():
                for window, target in val_loader:
                    window = window.to(device)
                    outputs = model(window) # Get raw outputs (logits)
                    _, predicted = torch.max(outputs.data, 1) # Get predicted class index
                    preds.extend(predicted.cpu().numpy())
                    actuals.extend(target.cpu().numpy())
            # Classification metrics
            accuracy = accuracy_score(actuals, preds)
            # Calculate precision, recall, f1 for each class and then average (e.g., weighted average)
            precision = precision_score(actuals, preds, average='weighted', zero_division=0)
            recall = recall_score(actuals, preds, average='weighted', zero_division=0)
            f1 = f1_score(actuals, preds, average='weighted', zero_division=0)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (weighted): {precision:.4f}")
            print(f"Recall (weighted): {recall:.4f}")
            print(f"F1 Score (weighted): {f1:.4f}")

            # Save the model using the registry
            from ml.src.models.model_registry import ModelRegistry # Import ModelRegistry
            registry = ModelRegistry()
            # Get basic metrics for saving
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            # Assuming the dataset object from get_dataloader has a scaler attribute
            # You might need to adjust how the scaler is accessed based on get_dataloader implementation
            scaler = train_loader.dataset.scaler if hasattr(train_loader.dataset, 'scaler') else None

            registry.save_model(
                model=model,
                symbol="BTC/USDT",
                metadata={
                    "model_type": "cnn_lstm", # Ensure consistent naming in metadata
                    "input_features": train_loader.dataset.feature_cols if hasattr(train_loader.dataset, 'feature_cols') else "Unknown",
                    "seq_len": seq_len,
                    "architecture": "CNNLSTM",
                    "num_classes": 3,
                    "training_mode": "standard"
                },
                preprocessor=scaler,
                metrics=metrics
            )

        except Exception as e:
            print(f"[ERROR] Exception during standard training: {e}")
            raise 