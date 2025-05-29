"""
Hyperparameter Tuning Module

This module provides utilities for optimizing model hyperparameters.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
import torch
from datetime import datetime
from sklearn.model_selection import ParameterGrid

# Import project modules
from ..models import ModelFactory
from .trainer import Trainer
from .evaluation import evaluate_model
from ..models.model_registry import get_registry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning for ML models"""
    
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        model_type: str = "lstm",
        input_dim: int = None,
        output_dim: int = None,
        seq_len: int = None,
        forecast_horizon: int = None,
        metric_name: str = "val_loss",
        higher_is_better: bool = False,
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        log_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        save_best_model: bool = True,
        n_trials: Optional[int] = None,
        random_search: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            param_grid: Dictionary of hyperparameters to tune
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            model_type: Type of model to tune
            input_dim: Input dimension
            output_dim: Output dimension
            seq_len: Sequence length
            forecast_horizon: Forecast horizon
            metric_name: Name of the metric to optimize
            higher_is_better: Whether higher metric values are better
            max_epochs: Maximum number of epochs to train each model
            early_stopping_patience: Patience for early stopping
            log_dir: Directory for logs
            experiment_name: Name for the experiment
            save_best_model: Whether to save the best model
            n_trials: Number of random trials (for random search)
            random_search: Whether to use random search instead of grid search
            random_state: Random state for reproducibility
        """
        self.param_grid = param_grid
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_type = model_type
        
        # Get data dimensions from dataloaders if not provided
        if input_dim is None:
            # Get from dataloader
            x_batch, _ = next(iter(train_dataloader))
            if len(x_batch.shape) == 3:  # (batch_size, seq_len, features)
                input_dim = x_batch.shape[2]
            else:
                input_dim = x_batch.shape[1]
                
        if output_dim is None:
            # Get from dataloader
            _, y_batch = next(iter(train_dataloader))
            if len(y_batch.shape) == 3:  # (batch_size, seq_len, features)
                output_dim = y_batch.shape[2]
            else:
                output_dim = 1
        
        if seq_len is None:
            # Get from dataloader
            x_batch, _ = next(iter(train_dataloader))
            if len(x_batch.shape) == 3:  # (batch_size, seq_len, features)
                seq_len = x_batch.shape[1]
            else:
                seq_len = 60  # Default
        
        if forecast_horizon is None:
            # Get from dataloader
            _, y_batch = next(iter(train_dataloader))
            if len(y_batch.shape) == 3:  # (batch_size, forecast_horizon, features)
                forecast_horizon = y_batch.shape[1]
            else:
                forecast_horizon = 5  # Default
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Set up experiment name and log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"tune_{model_type}_{timestamp}"
        self.log_dir = log_dir or os.path.join("logs", "tensorboard", self.experiment_name)
        
        self.save_best_model = save_best_model
        self.n_trials = n_trials
        self.random_search = random_search
        self.random_state = random_state
        
        # Initialize results storage
        self.results = []
        self.best_params = None
        self.best_metric = float('-inf') if higher_is_better else float('inf')
        self.best_model = None
        
        logger.info(f"Initialized hyperparameter tuning for {model_type}")
        logger.info(f"Parameter grid: {param_grid}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the hyperparameter tuning process.
        
        Returns:
            Dictionary containing results and best parameters
        """
        # Generate parameter combinations
        if self.random_search:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            param_list = list(ParameterGrid(self.param_grid))
            if self.n_trials is not None and self.n_trials < len(param_list):
                indices = np.random.choice(len(param_list), self.n_trials, replace=False)
                param_combinations = [param_list[i] for i in indices]
            else:
                param_combinations = param_list
        else:
            param_combinations = list(ParameterGrid(self.param_grid))
        
        n_combinations = len(param_combinations)
        logger.info(f"Running {n_combinations} hyperparameter combinations")
        
        # Run each combination
        for i, params in enumerate(param_combinations):
            logger.info(f"Trial {i+1}/{n_combinations}: {params}")
            
            # Extract model hyperparameters and training hyperparameters
            model_params = {}
            training_params = {}
            optimizer_params = {}
            
            for k, v in params.items():
                if k in ['hidden_dim', 'num_layers', 'dropout']:
                    model_params[k] = v
                elif k in ['learning_rate', 'weight_decay']:
                    optimizer_params[k] = v
                else:
                    training_params[k] = v
            
            # Create model
            model = ModelFactory.create_model(
                model_type=self.model_type,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                seq_len=self.seq_len,
                forecast_horizon=self.forecast_horizon,
                **model_params
            )
            
            # Set up trainer
            trial_name = f"{self.experiment_name}_trial_{i+1}"
            trial_log_dir = os.path.join(self.log_dir, f"trial_{i+1}")
            trial_checkpoints_dir = os.path.join("models", "checkpoints", trial_name)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                optimizer_kwargs=optimizer_params,
                log_dir=trial_log_dir,
                checkpoints_dir=trial_checkpoints_dir,
                experiment_name=trial_name
            )
            
            # Train model
            train_losses, val_losses = trainer.train(
                num_epochs=self.max_epochs,
                early_stopping_patience=self.early_stopping_patience,
                **training_params
            )
            
            # Evaluate model
            eval_results = evaluate_model(
                model=model,
                dataloader=self.val_dataloader,
                return_predictions=False
            )
            
            metrics = eval_results['metrics']
            
            # Record results
            trial_result = {
                'params': params,
                'metrics': metrics,
                'best_val_loss': trainer.best_val_loss,
                'epochs': len(train_losses),
                'trial': i + 1
            }
            
            self.results.append(trial_result)
            
            # Check if this is the best model
            metric_value = metrics.get(self.metric_name, trainer.best_val_loss)
            
            is_better = False
            if self.higher_is_better:
                is_better = metric_value > self.best_metric
            else:
                is_better = metric_value < self.best_metric
            
            if is_better:
                self.best_metric = metric_value
                self.best_params = params
                self.best_model = model
                
                logger.info(f"New best model: {self.metric_name} = {metric_value:.6f}")
                
                # Save the best model if requested
                if self.save_best_model:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    registry = get_registry()
                    registry.save_model(
                        model=model,
                        symbol='tuning_best',
                        metadata={
                            'tuning_experiment': self.experiment_name,
                            'params': params,
                            'metrics': metrics,
                            'best_val_loss': trainer.best_val_loss,
                            'epochs': len(train_losses),
                            'timestamp': timestamp
                        },
                        version=f"best_{timestamp}"
                    )
        
        # Sort results by the target metric
        sorted_results = sorted(
            self.results,
            key=lambda x: x['metrics'].get(self.metric_name, x['best_val_loss']),
            reverse=self.higher_is_better
        )
        
        # Prepare and save final results
        tuning_results = {
            'experiment': self.experiment_name,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_metric': {self.metric_name: self.best_metric},
            'results': sorted_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to file
        results_dir = os.path.join("models", "tuning_results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"{self.experiment_name}_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        
        logger.info(f"Hyperparameter tuning completed. Results saved to {results_path}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best {self.metric_name}: {self.best_metric:.6f}")
        
        return tuning_results


def tune_model(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model_type: str = "lstm",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter tuning.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        model_type: Type of model to tune
        param_grid: Dictionary of hyperparameters to tune
        **kwargs: Additional arguments for HyperparameterTuner
        
    Returns:
        Dictionary containing results and best parameters
    """
    # Default parameter grid if not provided
    if param_grid is None:
        if model_type.lower() == "lstm" or model_type.lower() == "gru":
            param_grid = {
                'hidden_dim': [64, 128, 256],
                'num_layers': [1, 2],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'weight_decay': [0, 1e-5]
            }
        elif model_type.lower() == "transformer":
            param_grid = {
                'hidden_dim': [64, 128, 256],
                'num_layers': [1, 2, 3],
                'dropout': [0.1, 0.2],
                'learning_rate': [0.001, 0.0005],
                'weight_decay': [0, 1e-5]
            }
        elif model_type.lower() == "cnn_lstm":
            param_grid = {
                'hidden_dim': [64, 128, 256],
                'num_layers': [1, 2],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.0005],
                'weight_decay': [0, 1e-5]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create and run tuner
    tuner = HyperparameterTuner(
        param_grid=param_grid,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model_type=model_type,
        **kwargs
    )
    
    return tuner.run()


def bayesian_optimize(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model_type: str = "lstm",
    n_trials: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform Bayesian optimization for hyperparameter tuning.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        model_type: Type of model to tune
        n_trials: Number of trials to run
        **kwargs: Additional arguments for HyperparameterTuner
        
    Returns:
        Dictionary containing results and best parameters
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna is required for Bayesian optimization. Install with 'pip install optuna'.")
        raise
    
    # Define the objective function
    def objective(trial):
        # Define hyperparameters to search
        if model_type.lower() in ["lstm", "gru"]:
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        elif model_type.lower() == "transformer":
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
            num_layers = trial.suggest_int('num_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        elif model_type.lower() == "cnn_lstm":
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
            num_layers = trial.suggest_int('num_layers', 1, 3)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Get dimensions from dataloaders
        x_batch, _ = next(iter(train_dataloader))
        _, y_batch = next(iter(train_dataloader))
        
        if len(x_batch.shape) == 3:  # (batch_size, seq_len, features)
            input_dim = x_batch.shape[2]
            seq_len = x_batch.shape[1]
        else:
            input_dim = x_batch.shape[1]
            seq_len = kwargs.get('seq_len', 60)
        
        if len(y_batch.shape) == 3:  # (batch_size, forecast_horizon, features)
            output_dim = y_batch.shape[2]
            forecast_horizon = y_batch.shape[1]
        else:
            output_dim = 1
            forecast_horizon = kwargs.get('forecast_horizon', 5)
        
        # Create model
        model = ModelFactory.create_model(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Set up trainer
        max_epochs = kwargs.get('max_epochs', 30)
        early_stopping_patience = kwargs.get('early_stopping_patience', 5)
        save_freq = kwargs.get('save_frequency', 5)
        
        trial_name = f"optuna_trial_{trial.number}"
        trial_log_dir = os.path.join(kwargs.get('log_dir', 'logs/tensorboard'), trial_name)
        trial_checkpoints_dir = os.path.join("models", "checkpoints", trial_name)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer_kwargs={'lr': learning_rate, 'weight_decay': weight_decay},
            log_dir=trial_log_dir,
            checkpoints_dir=trial_checkpoints_dir,
            experiment_name=trial_name
        )
        
        # Train model
        train_losses, val_losses = trainer.train(
            num_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            save_frequency=save_freq
        )
        
        # Return best validation loss
        return trainer.best_val_loss
    
    # Create a study
    metric_direction = "maximize" if kwargs.get('higher_is_better', False) else "minimize"
    study = optuna.create_study(direction=metric_direction)
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_trial = study.best_trial
    
    # Log results
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best value: {best_trial.value:.6f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = kwargs.get('experiment_name', f"optuna_{model_type}_{timestamp}")
    
    results_dir = os.path.join("models", "tuning_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{experiment_name}_results.json")
    
    results = {
        'experiment': experiment_name,
        'model_type': model_type,
        'best_params': best_params,
        'best_value': best_trial.value,
        'trials': [{
            'number': t.number,
            'params': t.params,
            'value': t.value
        } for t in study.trials],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Bayesian optimization completed. Results saved to {results_path}")
    
    return results 