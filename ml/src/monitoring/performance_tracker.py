"""
Performance Monitoring System

This module provides tools for tracking and monitoring model performance over time.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from threading import Lock
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_LOGS_DIR = os.environ.get("PERFORMANCE_LOGS_DIR", "logs/performance")


class PerformanceTracker:
    """
    Track and monitor model performance metrics over time.
    """
    
    def __init__(self, symbol: str, logs_dir: Optional[str] = None):
        """
        Initialize the performance tracker.
        
        Args:
            symbol: Trading symbol or model identifier
            logs_dir: Directory for performance logs
        """
        self.symbol = symbol
        self.logs_dir = logs_dir or DEFAULT_LOGS_DIR
        
        # Normalize symbol name for file paths
        self.symbol_name = symbol.replace("/", "_")
        
        # Create logs directory
        self.symbol_logs_dir = os.path.join(self.logs_dir, self.symbol_name)
        os.makedirs(self.symbol_logs_dir, exist_ok=True)
        
        # Log file paths
        self.metrics_log_path = os.path.join(self.symbol_logs_dir, "metrics.csv")
        self.predictions_log_path = os.path.join(self.symbol_logs_dir, "predictions.csv")
        self.drift_log_path = os.path.join(self.symbol_logs_dir, "drift.csv")
        
        # Reference distribution data
        self.reference_stats_path = os.path.join(self.symbol_logs_dir, "reference_stats.json")
        
        # Thread safety
        self.metrics_lock = Lock()
        self.predictions_lock = Lock()
        self.drift_lock = Lock()
        
        # Initialize logs if they don't exist
        self._initialize_logs()
        
        logger.info(f"Performance tracker initialized for {symbol}")
    
    def _initialize_logs(self):
        """Initialize log files if they don't exist"""
        # Create metrics log
        if not os.path.exists(self.metrics_log_path):
            metrics_df = pd.DataFrame(columns=[
                'timestamp', 'model_version', 'mse', 'rmse', 'mae', 'mape', 
                'r2', 'directional_accuracy', 'sharpe_ratio', 'custom_metrics'
            ])
            metrics_df.to_csv(self.metrics_log_path, index=False)
        
        # Create predictions log
        if not os.path.exists(self.predictions_log_path):
            predictions_df = pd.DataFrame(columns=[
                'timestamp', 'model_version', 'actual', 'predicted', 
                'horizon', 'features_json'
            ])
            predictions_df.to_csv(self.predictions_log_path, index=False)
        
        # Create drift log
        if not os.path.exists(self.drift_log_path):
            drift_df = pd.DataFrame(columns=[
                'timestamp', 'model_version', 'feature', 'p_value',
                'statistic', 'drift_detected', 'reference_mean', 'reference_std',
                'current_mean', 'current_std'
            ])
            drift_df.to_csv(self.drift_log_path, index=False)
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        model_version: str, 
        timestamp: Optional[str] = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            model_version: Version of the model used
            timestamp: Timestamp for the log entry (default: current time)
            custom_metrics: Additional custom metrics to log
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Extract standard metrics
        metrics_row = {
            'timestamp': timestamp,
            'model_version': model_version,
            'mse': metrics.get('mse', np.nan),
            'rmse': metrics.get('rmse', np.nan),
            'mae': metrics.get('mae', np.nan),
            'mape': metrics.get('mape', np.nan),
            'r2': metrics.get('r2', np.nan),
            'directional_accuracy': metrics.get('directional_accuracy', np.nan),
            'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
            'custom_metrics': json.dumps(custom_metrics or {})
        }
        
        # Create DataFrame for the new row
        new_row_df = pd.DataFrame([metrics_row])
        
        # Append to metrics log with thread safety
        with self.metrics_lock:
            try:
                # Load existing metrics
                metrics_df = pd.read_csv(self.metrics_log_path)
                
                # Append new row
                metrics_df = pd.concat([metrics_df, new_row_df], ignore_index=True)
                
                # Save back to file
                metrics_df.to_csv(self.metrics_log_path, index=False)
                
                logger.info(f"Logged metrics for {self.symbol} (version: {model_version})")
            except Exception as e:
                logger.error(f"Error logging metrics: {str(e)}")
    
    def log_prediction(
        self,
        actual: float,
        predicted: float,
        model_version: str,
        horizon: int = 1,
        features: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        check_for_drift: bool = True
    ) -> None:
        """
        Log a prediction and actual value pair.
        
        Args:
            actual: Actual observed value
            predicted: Model's predicted value
            model_version: Version of the model used
            horizon: Prediction horizon (e.g., 1-day ahead)
            features: Feature values used for prediction
            timestamp: Timestamp for the log entry (default: current time)
            check_for_drift: Whether to check for model drift
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create prediction entry
        prediction_row = {
            'timestamp': timestamp,
            'model_version': model_version,
            'actual': actual,
            'predicted': predicted,
            'horizon': horizon,
            'features_json': json.dumps(features or {})
        }
        
        # Create DataFrame for the new row
        new_row_df = pd.DataFrame([prediction_row])
        
        # Append to predictions log with thread safety
        with self.predictions_lock:
            try:
                # Load existing predictions
                predictions_df = pd.read_csv(self.predictions_log_path)
                
                # Append new row
                predictions_df = pd.concat([predictions_df, new_row_df], ignore_index=True)
                
                # Save back to file
                predictions_df.to_csv(self.predictions_log_path, index=False)
                
                logger.debug(f"Logged prediction for {self.symbol} (version: {model_version})")
            except Exception as e:
                logger.error(f"Error logging prediction: {str(e)}")
        
        # Check for drift if requested
        if check_for_drift and features:
            self.check_for_drift(features, model_version, timestamp)
    
    def set_reference_distribution(
        self,
        features: Dict[str, List[float]],
        model_version: str
    ) -> None:
        """
        Set reference distribution for drift detection.
        
        Args:
            features: Dictionary of feature names to feature values
            model_version: Version of the model
        """
        # Calculate statistics for each feature
        reference_stats = {
            'model_version': model_version,
            'timestamp': datetime.now().isoformat(),
            'features': {}
        }
        
        for feature_name, values in features.items():
            try:
                # Skip if values are not numeric
                if not all(isinstance(v, (int, float)) for v in values):
                    continue
                
                # Calculate statistics
                values_array = np.array(values)
                reference_stats['features'][feature_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'q25': float(np.percentile(values_array, 25)),
                    'q50': float(np.percentile(values_array, 50)),
                    'q75': float(np.percentile(values_array, 75)),
                    'count': len(values)
                }
            except Exception as e:
                logger.warning(f"Error calculating statistics for feature {feature_name}: {str(e)}")
        
        # Save reference statistics
        try:
            with open(self.reference_stats_path, 'w') as f:
                json.dump(reference_stats, f, indent=2)
            logger.info(f"Reference distribution set for {self.symbol} (version: {model_version})")
        except Exception as e:
            logger.error(f"Error saving reference distribution: {str(e)}")
    
    def check_for_drift(
        self,
        features: Dict[str, Any],
        model_version: str,
        timestamp: Optional[str] = None,
        window_size: int = 100,
        p_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check for drift in input features.
        
        Args:
            features: Current feature values
            model_version: Version of the model
            timestamp: Timestamp for the check (default: current time)
            window_size: Number of recent predictions to use for current distribution
            p_threshold: P-value threshold for drift detection
            
        Returns:
            Dictionary with drift detection results
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Check if reference distribution exists
        if not os.path.exists(self.reference_stats_path):
            logger.warning(f"No reference distribution found for {self.symbol}")
            return {'error': 'No reference distribution found'}
        
        # Load reference distribution
        try:
            with open(self.reference_stats_path, 'r') as f:
                reference_stats = json.load(f)
        except Exception as e:
            logger.error(f"Error loading reference distribution: {str(e)}")
            return {'error': f'Error loading reference distribution: {str(e)}'}
        
        # Load recent predictions for current distribution
        try:
            predictions_df = pd.read_csv(self.predictions_log_path)
            recent_predictions = predictions_df.tail(window_size)
        except Exception as e:
            logger.error(f"Error loading recent predictions: {str(e)}")
            return {'error': f'Error loading recent predictions: {str(e)}'}
        
        # Extract features from recent predictions
        current_features = {}
        drift_results = []
        
        for _, row in recent_predictions.iterrows():
            try:
                row_features = json.loads(row['features_json'])
                for name, value in row_features.items():
                    if isinstance(value, (int, float)):
                        if name not in current_features:
                            current_features[name] = []
                        current_features[name].append(value)
            except Exception as e:
                logger.warning(f"Error parsing features for prediction: {str(e)}")
        
        # Add current features
        for name, value in features.items():
            if isinstance(value, (int, float)):
                if name not in current_features:
                    current_features[name] = []
                current_features[name].append(value)
        
        # Check drift for each feature
        ref_features = reference_stats.get('features', {})
        
        for name, values in current_features.items():
            if name in ref_features and len(values) >= 30:  # Need enough samples for statistical testing
                try:
                    # Calculate current statistics
                    current_array = np.array(values)
                    current_mean = float(np.mean(current_array))
                    current_std = float(np.std(current_array))
                    
                    # Reference statistics
                    ref_mean = ref_features[name]['mean']
                    ref_std = ref_features[name]['std']
                    
                    # Perform KS test
                    # Create reference distribution using mean and std
                    ref_array = np.random.normal(
                        loc=ref_mean, 
                        scale=max(ref_std, 1e-5),  # Avoid zero std
                        size=len(current_array)
                    )
                    
                    # Calculate drift
                    statistic, p_value = stats.ks_2samp(current_array, ref_array)
                    drift_detected = p_value < p_threshold
                    
                    # Create drift record
                    drift_record = {
                        'timestamp': timestamp,
                        'model_version': model_version,
                        'feature': name,
                        'p_value': float(p_value),
                        'statistic': float(statistic),
                        'drift_detected': drift_detected,
                        'reference_mean': ref_mean,
                        'reference_std': ref_std,
                        'current_mean': current_mean,
                        'current_std': current_std
                    }
                    
                    drift_results.append(drift_record)
                    
                    # Log warning if drift detected
                    if drift_detected:
                        logger.warning(f"Drift detected in feature {name} for {self.symbol} (p={p_value:.4f})")
                except Exception as e:
                    logger.warning(f"Error checking drift for feature {name}: {str(e)}")
        
        # Log drift results
        if drift_results:
            drift_df = pd.DataFrame(drift_results)
            with self.drift_lock:
                try:
                    # Load existing drift log
                    if os.path.exists(self.drift_log_path):
                        existing_df = pd.read_csv(self.drift_log_path)
                        updated_df = pd.concat([existing_df, drift_df], ignore_index=True)
                    else:
                        updated_df = drift_df
                    
                    # Save back to file
                    updated_df.to_csv(self.drift_log_path, index=False)
                except Exception as e:
                    logger.error(f"Error logging drift results: {str(e)}")
        
        return {'drift_results': drift_results}
    
    def get_metrics_history(
        self,
        model_version: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical performance metrics.
        
        Args:
            model_version: Filter by model version
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            limit: Maximum number of entries to return
            
        Returns:
            DataFrame with historical metrics
        """
        try:
            # Load metrics
            metrics_df = pd.read_csv(self.metrics_log_path)
            
            # Apply filters
            if model_version is not None:
                metrics_df = metrics_df[metrics_df['model_version'] == model_version]
            
            if start_time is not None:
                metrics_df = metrics_df[metrics_df['timestamp'] >= start_time]
            
            if end_time is not None:
                metrics_df = metrics_df[metrics_df['timestamp'] <= end_time]
            
            # Sort by timestamp
            metrics_df = metrics_df.sort_values('timestamp', ascending=False)
            
            # Apply limit
            if limit is not None:
                metrics_df = metrics_df.head(limit)
            
            return metrics_df
        
        except Exception as e:
            logger.error(f"Error getting metrics history: {str(e)}")
            return pd.DataFrame()
    
    def get_predictions_history(
        self,
        model_version: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical prediction data.
        
        Args:
            model_version: Filter by model version
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            limit: Maximum number of entries to return
            
        Returns:
            DataFrame with historical predictions
        """
        try:
            # Load predictions
            predictions_df = pd.read_csv(self.predictions_log_path)
            
            # Apply filters
            if model_version is not None:
                predictions_df = predictions_df[predictions_df['model_version'] == model_version]
            
            if start_time is not None:
                predictions_df = predictions_df[predictions_df['timestamp'] >= start_time]
            
            if end_time is not None:
                predictions_df = predictions_df[predictions_df['timestamp'] <= end_time]
            
            # Sort by timestamp
            predictions_df = predictions_df.sort_values('timestamp', ascending=False)
            
            # Apply limit
            if limit is not None:
                predictions_df = predictions_df.head(limit)
            
            return predictions_df
        
        except Exception as e:
            logger.error(f"Error getting predictions history: {str(e)}")
            return pd.DataFrame()
    
    def calculate_rolling_metrics(
        self,
        window_size: int = 100,
        model_version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            window_size: Size of the rolling window
            model_version: Filter by model version
            
        Returns:
            DataFrame with rolling metrics
        """
        try:
            # Load predictions
            predictions_df = pd.read_csv(self.predictions_log_path)
            
            # Apply filters
            if model_version is not None:
                predictions_df = predictions_df[predictions_df['model_version'] == model_version]
            
            # Sort by timestamp
            predictions_df = predictions_df.sort_values('timestamp')
            
            # Convert timestamp to datetime for proper rolling window
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            
            # Initialize rolling metrics DataFrame
            rolling_metrics = pd.DataFrame({
                'timestamp': predictions_df['timestamp']
            })
            
            # Calculate error metrics
            predictions_df['error'] = predictions_df['predicted'] - predictions_df['actual']
            predictions_df['squared_error'] = predictions_df['error'] ** 2
            predictions_df['absolute_error'] = np.abs(predictions_df['error'])
            
            # Calculate direction
            predictions_df['actual_direction'] = predictions_df['actual'].diff() > 0
            predictions_df['predicted_direction'] = predictions_df['predicted'].diff() > 0
            predictions_df['direction_correct'] = predictions_df['actual_direction'] == predictions_df['predicted_direction']
            
            # Calculate rolling metrics
            rolling_metrics['rmse'] = np.sqrt(
                predictions_df['squared_error'].rolling(window=window_size).mean()
            )
            
            rolling_metrics['mae'] = predictions_df['absolute_error'].rolling(window=window_size).mean()
            
            # Handle potential division by zero in MAPE calculation
            predictions_df['mape_value'] = 100 * predictions_df['absolute_error'] / np.maximum(
                np.abs(predictions_df['actual']), 1e-10
            )
            rolling_metrics['mape'] = predictions_df['mape_value'].rolling(window=window_size).mean()
            
            # Directional accuracy
            rolling_metrics['directional_accuracy'] = predictions_df['direction_correct'].rolling(window=window_size).mean()
            
            # Calculate rolling Sharpe ratio if actual data is present
            if not predictions_df['actual'].isnull().all():
                # Simple returns based on actual values
                predictions_df['actual_returns'] = predictions_df['actual'].pct_change()
                
                # Trading returns based on predicted direction
                predictions_df['trading_returns'] = np.where(
                    predictions_df['predicted_direction'],
                    predictions_df['actual_returns'],
                    -predictions_df['actual_returns']
                )
                
                # Calculate rolling Sharpe ratio (annualized)
                rolling_metrics['sharpe_ratio'] = (
                    predictions_df['trading_returns'].rolling(window=window_size).mean() / 
                    predictions_df['trading_returns'].rolling(window=window_size).std()
                ) * np.sqrt(252)  # Annualized
            
            # Drop NaN values (from the rolling window)
            rolling_metrics = rolling_metrics.dropna()
            
            return rolling_metrics
        
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {str(e)}")
            return pd.DataFrame()
    
    def get_drift_history(
        self,
        feature: Optional[str] = None,
        model_version: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get historical drift detection data.
        
        Args:
            feature: Filter by feature name
            model_version: Filter by model version
            start_time: Start time for filtering (ISO format)
            end_time: End time for filtering (ISO format)
            limit: Maximum number of entries to return
            
        Returns:
            DataFrame with drift history
        """
        try:
            # Check if drift log exists
            if not os.path.exists(self.drift_log_path):
                return pd.DataFrame()
            
            # Load drift log
            drift_df = pd.read_csv(self.drift_log_path)
            
            # Apply filters
            if feature is not None:
                drift_df = drift_df[drift_df['feature'] == feature]
            
            if model_version is not None:
                drift_df = drift_df[drift_df['model_version'] == model_version]
            
            if start_time is not None:
                drift_df = drift_df[drift_df['timestamp'] >= start_time]
            
            if end_time is not None:
                drift_df = drift_df[drift_df['timestamp'] <= end_time]
            
            # Sort by timestamp
            drift_df = drift_df.sort_values('timestamp', ascending=False)
            
            # Apply limit
            if limit is not None:
                drift_df = drift_df.head(limit)
            
            return drift_df
        
        except Exception as e:
            logger.error(f"Error getting drift history: {str(e)}")
            return pd.DataFrame()
    
    def generate_performance_report(
        self,
        model_version: Optional[str] = None,
        output_dir: Optional[str] = None,
        last_n_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a performance report.
        
        Args:
            model_version: Filter by model version
            output_dir: Directory to save report artifacts
            last_n_days: Filter to include only the last N days
            
        Returns:
            Dictionary with report summary
        """
        try:
            # Load metrics and predictions
            metrics_df = pd.read_csv(self.metrics_log_path)
            predictions_df = pd.read_csv(self.predictions_log_path)
            
            # Filter by model version
            if model_version is not None:
                metrics_df = metrics_df[metrics_df['model_version'] == model_version]
                predictions_df = predictions_df[predictions_df['model_version'] == model_version]
            
            # Filter by time if needed
            if last_n_days is not None:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=last_n_days)
                
                start_date_str = start_date.isoformat()
                
                metrics_df = metrics_df[metrics_df['timestamp'] >= start_date_str]
                predictions_df = predictions_df[predictions_df['timestamp'] >= start_date_str]
            
            # Check if we have data
            if len(metrics_df) == 0 and len(predictions_df) == 0:
                return {
                    'error': 'No data available for the specified filters'
                }
            
            # Convert timestamps to datetime for better handling
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            
            # Calculate report summary
            summary = {
                'symbol': self.symbol,
                'model_version': model_version or 'all',
                'report_generated_at': datetime.now().isoformat(),
                'data_range': {}
            }
            
            # Set date range
            if len(metrics_df) > 0:
                summary['data_range']['metrics_start'] = metrics_df['timestamp'].min().isoformat()
                summary['data_range']['metrics_end'] = metrics_df['timestamp'].max().isoformat()
                summary['metrics_count'] = len(metrics_df)
            
            if len(predictions_df) > 0:
                summary['data_range']['predictions_start'] = predictions_df['timestamp'].min().isoformat()
                summary['data_range']['predictions_end'] = predictions_df['timestamp'].max().isoformat()
                summary['predictions_count'] = len(predictions_df)
            
            # Calculate prediction error statistics
            if len(predictions_df) > 0:
                predictions_df['error'] = predictions_df['predicted'] - predictions_df['actual']
                predictions_df['squared_error'] = predictions_df['error'] ** 2
                predictions_df['absolute_error'] = np.abs(predictions_df['error'])
                
                # For directional accuracy
                predictions_df['actual_diff'] = predictions_df['actual'].diff()
                predictions_df['predicted_diff'] = predictions_df['predicted'].diff()
                predictions_df['correct_direction'] = (
                    (predictions_df['actual_diff'] > 0) & (predictions_df['predicted_diff'] > 0) |
                    (predictions_df['actual_diff'] < 0) & (predictions_df['predicted_diff'] < 0)
                )
                
                # Calculate general error statistics
                summary['error_statistics'] = {
                    'mse': predictions_df['squared_error'].mean(),
                    'rmse': np.sqrt(predictions_df['squared_error'].mean()),
                    'mae': predictions_df['absolute_error'].mean(),
                    'median_ae': predictions_df['absolute_error'].median(),
                    'max_error': predictions_df['error'].abs().max(),
                    'directional_accuracy': predictions_df['correct_direction'].mean()
                }
                
                # Calculate error distribution
                error_percentiles = np.percentile(predictions_df['error'], [5, 25, 50, 75, 95])
                summary['error_distribution'] = {
                    'p5': error_percentiles[0],
                    'p25': error_percentiles[1],
                    'p50': error_percentiles[2],
                    'p75': error_percentiles[3],
                    'p95': error_percentiles[4]
                }
            
            # Check for drift
            try:
                if os.path.exists(self.drift_log_path):
                    drift_df = pd.read_csv(self.drift_log_path)
                    
                    # Filter by model version
                    if model_version is not None:
                        drift_df = drift_df[drift_df['model_version'] == model_version]
                    
                    # Filter by time if needed
                    if last_n_days is not None:
                        drift_df['timestamp'] = pd.to_datetime(drift_df['timestamp'])
                        drift_df = drift_df[drift_df['timestamp'] >= start_date]
                    
                    if len(drift_df) > 0:
                        # Count drift occurrences by feature
                        drift_counts = drift_df[drift_df['drift_detected']].groupby('feature').size()
                        
                        summary['drift_analysis'] = {
                            'total_checks': len(drift_df),
                            'total_drift_detections': len(drift_df[drift_df['drift_detected']]),
                            'drift_rate': len(drift_df[drift_df['drift_detected']]) / len(drift_df) if len(drift_df) > 0 else 0,
                            'features_with_drift': drift_counts.to_dict()
                        }
            except Exception as e:
                logger.warning(f"Error analyzing drift: {str(e)}")
            
            # Calculate rolling metrics
            try:
                rolling_metrics = self.calculate_rolling_metrics(window_size=30, model_version=model_version)
                if len(rolling_metrics) > 0:
                    summary['rolling_metrics'] = {
                        'latest_rmse': rolling_metrics['rmse'].iloc[-1],
                        'latest_mae': rolling_metrics['mae'].iloc[-1],
                        'latest_directional_accuracy': rolling_metrics['directional_accuracy'].iloc[-1],
                        'rmse_trend': 'improving' if rolling_metrics['rmse'].iloc[-1] < rolling_metrics['rmse'].iloc[0] else 'worsening',
                        'directional_accuracy_trend': 'improving' if rolling_metrics['directional_accuracy'].iloc[-1] > rolling_metrics['directional_accuracy'].iloc[0] else 'worsening'
                    }
                    
                    if 'sharpe_ratio' in rolling_metrics.columns:
                        summary['rolling_metrics']['latest_sharpe_ratio'] = rolling_metrics['sharpe_ratio'].iloc[-1]
            except Exception as e:
                logger.warning(f"Error calculating rolling metrics: {str(e)}")
            
            # Generate visualizations if output directory is provided
            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Save prediction vs actual plot
                    if len(predictions_df) > 0:
                        plt.figure(figsize=(12, 6))
                        
                        # Get last 100 predictions for clarity
                        plot_df = predictions_df.sort_values('timestamp').tail(100)
                        
                        plt.plot(plot_df['timestamp'], plot_df['actual'], 'b-', label='Actual')
                        plt.plot(plot_df['timestamp'], plot_df['predicted'], 'r--', label='Predicted')
                        
                        plt.title(f'Actual vs Predicted Values for {self.symbol}')
                        plt.xlabel('Time')
                        plt.ylabel('Value')
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        plt.savefig(output_path / 'actual_vs_predicted.png', dpi=300)
                        plt.close()
                        
                        # Save error distribution plot
                        plt.figure(figsize=(10, 6))
                        plt.hist(predictions_df['error'], bins=30, alpha=0.7)
                        plt.axvline(x=0, color='r', linestyle='--')
                        
                        plt.title(f'Prediction Error Distribution for {self.symbol}')
                        plt.xlabel('Error (Predicted - Actual)')
                        plt.ylabel('Frequency')
                        plt.tight_layout()
                        
                        plt.savefig(output_path / 'error_distribution.png', dpi=300)
                        plt.close()
                    
                    # Save rolling metrics plot
                    if 'rolling_metrics' in summary:
                        plt.figure(figsize=(12, 8))
                        
                        # Create subplots
                        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
                        
                        # RMSE
                        axes[0].plot(rolling_metrics['timestamp'], rolling_metrics['rmse'], 'b-')
                        axes[0].set_title('Rolling RMSE')
                        axes[0].set_ylabel('RMSE')
                        axes[0].grid(True, alpha=0.3)
                        
                        # MAE
                        axes[1].plot(rolling_metrics['timestamp'], rolling_metrics['mae'], 'g-')
                        axes[1].set_title('Rolling MAE')
                        axes[1].set_ylabel('MAE')
                        axes[1].grid(True, alpha=0.3)
                        
                        # Directional Accuracy
                        axes[2].plot(rolling_metrics['timestamp'], rolling_metrics['directional_accuracy'], 'r-')
                        axes[2].set_title('Rolling Directional Accuracy')
                        axes[2].set_ylabel('Accuracy')
                        axes[2].set_ylim(0, 1)
                        axes[2].grid(True, alpha=0.3)
                        
                        # Set common x-axis label
                        plt.xlabel('Time')
                        plt.xticks(rotation=45)
                        
                        plt.tight_layout()
                        plt.savefig(output_path / 'rolling_metrics.png', dpi=300)
                        plt.close()
                    
                    # Save drift analysis if available
                    if 'drift_analysis' in summary and os.path.exists(self.drift_log_path):
                        drift_df = pd.read_csv(self.drift_log_path)
                        
                        # Filter as needed
                        if model_version is not None:
                            drift_df = drift_df[drift_df['model_version'] == model_version]
                        
                        if last_n_days is not None:
                            drift_df['timestamp'] = pd.to_datetime(drift_df['timestamp'])
                            drift_df = drift_df[drift_df['timestamp'] >= start_date]
                        
                        if len(drift_df) > 0:
                            # Plot drift detections over time
                            plt.figure(figsize=(12, 6))
                            
                            # Group by timestamp and feature, count drift detections
                            drift_df['timestamp'] = pd.to_datetime(drift_df['timestamp'])
                            drift_counts = drift_df[drift_df['drift_detected']].groupby(
                                [pd.Grouper(key='timestamp', freq='D'), 'feature']
                            ).size().unstack(fill_value=0)
                            
                            # Plot drift counts
                            drift_counts.plot(kind='bar', stacked=True, ax=plt.gca())
                            
                            plt.title(f'Drift Detections Over Time for {self.symbol}')
                            plt.xlabel('Date')
                            plt.ylabel('Number of Drift Detections')
                            plt.legend(title='Feature')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            
                            plt.savefig(output_path / 'drift_analysis.png', dpi=300)
                            plt.close()
                    
                    # Save report as JSON
                    with open(output_path / 'performance_report.json', 'w') as f:
                        json.dump(summary, f, indent=2)
                    
                    # Also save summary as text
                    with open(output_path / 'summary.txt', 'w') as f:
                        f.write(f"Performance Report for {self.symbol}\n")
                        f.write(f"Generated: {summary['report_generated_at']}\n\n")
                        
                        f.write("Data Range:\n")
                        if 'metrics_start' in summary['data_range']:
                            f.write(f"  Metrics: {summary['data_range']['metrics_start']} to {summary['data_range']['metrics_end']}\n")
                        if 'predictions_start' in summary['data_range']:
                            f.write(f"  Predictions: {summary['data_range']['predictions_start']} to {summary['data_range']['predictions_end']}\n")
                        
                        if 'error_statistics' in summary:
                            f.write("\nError Statistics:\n")
                            for key, value in summary['error_statistics'].items():
                                f.write(f"  {key}: {value:.4f}\n")
                        
                        if 'drift_analysis' in summary:
                            f.write("\nDrift Analysis:\n")
                            f.write(f"  Total Checks: {summary['drift_analysis']['total_checks']}\n")
                            f.write(f"  Drift Detections: {summary['drift_analysis']['total_drift_detections']}\n")
                            f.write(f"  Drift Rate: {summary['drift_analysis']['drift_rate']:.2%}\n")
                
                except Exception as e:
                    logger.error(f"Error generating visualizations: {str(e)}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {'error': str(e)}


# Global registry of performance trackers
_trackers = {}
_trackers_lock = Lock()

def get_tracker(symbol: str) -> PerformanceTracker:
    """
    Get or create a performance tracker for a symbol.
    
    Args:
        symbol: Trading symbol or model identifier
        
    Returns:
        Performance tracker instance
    """
    with _trackers_lock:
        if symbol not in _trackers:
            _trackers[symbol] = PerformanceTracker(symbol)
        return _trackers[symbol] 