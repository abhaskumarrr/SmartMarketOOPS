"""
Model Registry

This module provides utilities for managing model versioning, storage, and retrieval.
"""

import os
import json
import shutil
import pickle
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..models import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default registry path
DEFAULT_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "models/registry")


class ModelRegistry:
    """
    Registry for managing ML models with versioning support.

    This class handles saving, loading, and tracking model versions.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the model registry.

        Args:
            registry_path: Path to the model registry directory
        """
        self.registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model registry initialized at {self.registry_path}")

    def save_model(
        self,
        model: torch.nn.Module,
        symbol: str,
        metadata: Dict[str, Any],
        preprocessor: Optional[Any] = None,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model to the registry with versioning.

        Args:
            model: The PyTorch model to save
            symbol: Trading symbol or model identifier
            metadata: Model metadata (training params, architecture, etc.)
            preprocessor: Optional preprocessor used with the model
            version: Optional specific version to use (default: timestamp-based)
            metrics: Optional performance metrics
            artifacts: Optional additional artifacts to save

        Returns:
            The version string of the saved model
        """
        # Normalize symbol name for file paths (remove '/' and '_')
        # Ensure consistency with the naming convention used in model_service.py
        # symbol_name = symbol.replace("/", "").replace("_", "") # Original: removed _ as well
        symbol_name = symbol.replace("/", "") # Only remove /, matching model_service.py

        # Create version string if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"

        # Create model directory
        model_dir = self.registry_path / symbol_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model checkpoint
        model_path = model_dir / "model.pt"

        # Get model configuration
        model_config = {}
        for attr in ['input_dim', 'output_dim', 'hidden_dim', 'num_layers',
                    'dropout', 'seq_len', 'forecast_horizon']:
            if hasattr(model, attr):
                model_config[attr] = getattr(model, attr)

        # Determine model type to save in checkpoint
        saved_model_type = model.__class__.__name__
        if saved_model_type == 'CNNLSTMModel':
            saved_model_type = 'cnn_lstm' # Ensure consistent naming

        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'model_type': saved_model_type, # Use the determined model type
        }

        # Print the model type being saved
        print(f"ModelRegistry saving model_type to checkpoint: {saved_model_type}")

        # Save the model
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")

        # Also save as best.pt for compatibility with existing code
        torch.save(checkpoint, model_dir / "best.pt")

        # Save preprocessor if provided
        if preprocessor is not None:
            preprocessor_path = model_dir / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            logger.info(f"Preprocessor saved to {preprocessor_path}")

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        # Save metrics if provided
        if metrics is not None:
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")

        # Save additional artifacts if provided
        if artifacts is not None:
            artifacts_dir = model_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            for name, artifact in artifacts.items():
                if isinstance(artifact, (dict, list)):
                    # Save JSON serializable artifacts
                    artifact_path = artifacts_dir / f"{name}.json"
                    with open(artifact_path, 'w') as f:
                        json.dump(artifact, f, indent=2)
                elif isinstance(artifact, (str, bytes, Path)):
                    # Save file artifacts
                    if isinstance(artifact, (str, Path)):
                        src_path = Path(artifact)
                        if src_path.exists():
                            dst_path = artifacts_dir / src_path.name
                            shutil.copy(src_path, dst_path)
                            logger.info(f"Artifact {name} copied to {dst_path}")
                    else:
                        # Save bytes
                        artifact_path = artifacts_dir / f"{name}.bin"
                        with open(artifact_path, 'wb') as f:
                            f.write(artifact)
                else:
                    try:
                        # Try to pickle the artifact
                        artifact_path = artifacts_dir / f"{name}.pkl"
                        with open(artifact_path, 'wb') as f:
                            pickle.dump(artifact, f)
                        logger.info(f"Artifact {name} saved to {artifact_path}")
                    except Exception as e:
                        logger.warning(f"Could not save artifact {name}: {str(e)}")

        # Update versions index file
        self._update_versions_index(symbol_name, version, metrics)

        return version

    def _update_versions_index(self, symbol_name: str, version: str, metrics: Optional[Dict[str, float]] = None):
        """
        Update the versions index file for a symbol.

        Args:
            symbol_name: Normalized symbol name
            version: Version identifier
            metrics: Optional performance metrics
        """
        # Path to versions index file
        index_path = self.registry_path / symbol_name / "versions_index.json"

        # Initialize or load versions index
        if index_path.exists():
            with open(index_path, 'r') as f:
                versions_index = json.load(f)
        else:
            versions_index = {"versions": []}

        # Get version metadata
        metadata_path = self.registry_path / symbol_name / version / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Create version entry
        version_entry = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_type": metadata.get("model_type", "unknown"),
            "description": metadata.get("description", ""),
        }

        # Add metrics if provided
        if metrics is not None:
            version_entry["metrics"] = metrics

        # Update versions list
        versions_index["versions"].append(version_entry)

        # Sort versions by creation time (newest first)
        versions_index["versions"].sort(
            key=lambda v: v.get("created_at", ""),
            reverse=True
        )

        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(versions_index, f, indent=2)

    def load_model(
        self,
        symbol: str,
        version: Optional[str] = None,
        return_metadata: bool = False,
        return_preprocessor: bool = False,
        device: Optional[str] = None
    ) -> Union[torch.nn.Module, Tuple]:
        """
        Load a model from the registry.

        Args:
            symbol: Trading symbol or model identifier
            version: Specific version to load (default: latest)
            return_metadata: Whether to return metadata with the model
            return_preprocessor: Whether to return the preprocessor with the model
            device: Device to load the model on (default: CPU)

        Returns:
            The loaded model or a tuple of (model, [metadata], [preprocessor])
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "")

        # Determine model directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No models found for {symbol}")

        # Find the specific version or latest
        if version is None:
            versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "artifacts"]
            if not versions:
                raise FileNotFoundError(f"No model versions found for {symbol}")
            version = sorted(versions)[-1]

        version_dir = model_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found for {symbol}")

        # Load model
        model_path = version_dir / "best.pt"
        if not model_path.exists():
            model_path = version_dir / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found in {version_dir}")

        # Map model to device
        device_map = torch.device('cpu' if device is None else device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device_map, weights_only=False)

        # Get model config and type
        model_config = checkpoint.get('model_config', {})
        model_type = checkpoint.get('model_type', '')
        # Normalize model_type to match ModelFactory keys
        model_type_map = {
            'lstm': 'lstm',
            'lstmmodel': 'lstm',
            'LSTMModel': 'lstm',
            'gru': 'gru',
            'grumodel': 'gru',
            'GRUModel': 'gru',
            'transformer': 'transformer',
            'transformermodel': 'transformer',
            'TransformerModel': 'transformer',
            'cnn_lstm': 'cnn_lstm',
            'cnnlstmmodel': 'cnn_lstm',
            'CNNLSTMModel': 'cnn_lstm',
        }
        model_type_key = str(model_type).lower()
        model_type = model_type_map.get(model_type_key, model_type_key)
        if not model_type and 'model_type' in model_config:
            model_type = model_config['model_type'].lower()
            model_type = model_type_map.get(model_type, model_type)
        # Default to LSTM if type not found
        if not model_type:
            model_type = 'lstm'
            logger.warning(f"Model type not found in checkpoint, defaulting to {model_type}")

        # Print the normalized model_type before passing to ModelFactory
        print(f"model_registry.load_model passing normalized model_type to ModelFactory: {model_type}")

        # Create model instance
        model = ModelFactory.create_model(
            model_type=model_type,
            input_dim=model_config.get('input_dim', 10),
            output_dim=model_config.get('output_dim', 1),
            seq_len=model_config.get('seq_len', 60),
            forecast_horizon=model_config.get('forecast_horizon', 5),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 2),
            device=device
        )

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        results = [model]

        # Load metadata if requested
        if return_metadata:
            metadata_path = version_dir / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            results.append(metadata)

        # Load preprocessor if requested
        if return_preprocessor:
            preprocessor = None
            preprocessor_path = version_dir / "preprocessor.pkl"
            if preprocessor_path.exists():
                with open(preprocessor_path, 'rb') as f:
                    preprocessor = pickle.load(f)
            results.append(preprocessor)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def get_versions(self, symbol: str) -> List[str]:
        """
        Get all available versions for a symbol.

        Args:
            symbol: Trading symbol or model identifier

        Returns:
            List of available versions
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "")

        # Get directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            return []

        # Get versions
        versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "artifacts"]
        return sorted(versions)

    def get_symbols(self) -> List[str]:
        """
        Get all available symbols in the registry.

        Returns:
            List of available symbols
        """
        if not self.registry_path.exists():
            return []

        # Get all directories
        symbols = [d.name for d in self.registry_path.iterdir() if d.is_dir()]

        # Convert back to original format
        symbols = [s.replace("_", "/") for s in symbols]
        return sorted(symbols)

    def get_metadata(self, symbol: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a specific model version.

        Args:
            symbol: Trading symbol or model identifier
            version: Specific version to get metadata for (default: latest)

        Returns:
            Model metadata
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")

        # Determine model directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No models found for {symbol}")

        # Find the specific version or latest
        if version is None:
            versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "artifacts"]
            if not versions:
                raise FileNotFoundError(f"No model versions found for {symbol}")
            version = sorted(versions)[-1]

        version_dir = model_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found for {symbol}")

        # Load metadata
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def get_metrics(self, symbol: str, version: Optional[str] = None) -> Dict[str, float]:
        """
        Get performance metrics for a specific model version.

        Args:
            symbol: Trading symbol or model identifier
            version: Specific version to get metrics for (default: latest)

        Returns:
            Model metrics
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")

        # Determine model directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No models found for {symbol}")

        # Find the specific version or latest
        if version is None:
            versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "artifacts"]
            if not versions:
                raise FileNotFoundError(f"No model versions found for {symbol}")
            version = sorted(versions)[-1]

        version_dir = model_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found for {symbol}")

        # Load metrics
        metrics_path = version_dir / "metrics.json"
        if not metrics_path.exists():
            return {}

        with open(metrics_path, 'r') as f:
            return json.load(f)

    def compare_models(
        self,
        symbol: str,
        versions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple model versions for a given symbol.

        Args:
            symbol: Trading symbol or model identifier
            versions: List of versions to compare (default: all versions)
            metrics: List of metric names to compare (default: all metrics)
            output_path: Path to save comparison results (default: None)

        Returns:
            Dictionary with comparison results
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")

        # Determine model directory
        model_dir = self.registry_path / symbol_name
        if not model_dir.exists():
            raise FileNotFoundError(f"No models found for {symbol}")

        # Get all versions if not specified
        if versions is None:
            versions = self.get_versions(symbol)

        if not versions:
            raise ValueError(f"No versions available for {symbol}")

        # Container for results
        results = {
            "symbol": symbol,
            "versions": [],
            "metrics_comparison": {},
            "metadata_comparison": {}
        }

        # Collect metrics and metadata for each version
        for version in versions:
            try:
                # Get metrics
                version_metrics = self.get_metrics(symbol, version)

                # Get metadata
                version_metadata = self.get_metadata(symbol, version)

                # Store in results
                version_data = {
                    "version": version,
                    "metrics": version_metrics,
                    "metadata": {
                        "model_type": version_metadata.get("model_type", "unknown"),
                        "created_at": version_metadata.get("created_at", "unknown"),
                        "description": version_metadata.get("description", "")
                    }
                }

                results["versions"].append(version_data)
            except Exception as e:
                logger.warning(f"Error loading data for version {version}: {str(e)}")

        # Filter metrics if specified
        if metrics is not None:
            for version_data in results["versions"]:
                filtered_metrics = {}
                for metric in metrics:
                    if metric in version_data["metrics"]:
                        filtered_metrics[metric] = version_data["metrics"][metric]
                version_data["metrics"] = filtered_metrics

        # Create metrics comparison
        all_metrics = set()
        for version_data in results["versions"]:
            all_metrics.update(version_data["metrics"].keys())

        # Compare each metric across versions
        for metric in all_metrics:
            metric_values = []
            for version_data in results["versions"]:
                if metric in version_data["metrics"]:
                    metric_values.append({
                        "version": version_data["version"],
                        "value": version_data["metrics"][metric]
                    })

            if metric_values:
                # Sort by value (best first, assuming lower is better for most metrics)
                sorted_values = sorted(metric_values, key=lambda x: x["value"])

                # Reverse sort order for metrics where higher is better
                if any(metric.endswith(m) for m in ["accuracy", "r2", "sharpe_ratio", "directional_accuracy", "win_rate"]):
                    sorted_values = sorted(metric_values, key=lambda x: x["value"], reverse=True)

                results["metrics_comparison"][metric] = sorted_values

        # Create visualizations if output path is provided
        if output_path is not None:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate comparison report
            self._generate_comparison_visualizations(results, output_dir)

            # Save JSON results
            results_path = output_dir / "comparison_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        return results

    def _generate_comparison_visualizations(self, results: Dict[str, Any], output_dir: Path):
        """
        Generate visualizations for model comparison.

        Args:
            results: Comparison results dictionary
            output_dir: Directory to save visualizations
        """
        symbol = results["symbol"]
        versions = [v["version"] for v in results["versions"]]

        # Create metrics comparison plots
        for metric, values in results["metrics_comparison"].items():
            # Create figure
            plt.figure(figsize=(10, 6))

            # Extract data
            metric_versions = [v["version"] for v in values]
            metric_values = [v["value"] for v in values]

            # Create bar chart
            bars = plt.bar(metric_versions, metric_values)

            # Add value labels on top of bars
            for bar, value in zip(bars, metric_values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f'{value:.4f}',
                    ha='center',
                    va='bottom',
                    rotation=45 if len(metric_versions) > 4 else 0,
                    fontsize=8
                )

            # Add labels and title
            plt.xlabel('Model Version')
            plt.ylabel(metric)
            plt.title(f'{metric} Comparison for {symbol}')

            # Adjust layout and save
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"{metric}_comparison.png", dpi=300)
            plt.close()

        # Create metrics table
        metrics_df = pd.DataFrame()
        for version_data in results["versions"]:
            version = version_data["version"]
            metrics = version_data["metrics"]

            # Add to DataFrame
            version_df = pd.DataFrame([metrics], index=[version])
            metrics_df = pd.concat([metrics_df, version_df])

        # Save metrics table
        if not metrics_df.empty:
            # Save as CSV
            metrics_df.to_csv(output_dir / "metrics_comparison.csv")

            # Create visualization of metrics table
            plt.figure(figsize=(12, len(metrics_df) * 0.5 + 2))

            # Create table
            table = plt.table(
                cellText=metrics_df.values.round(4),
                rowLabels=metrics_df.index,
                colLabels=metrics_df.columns,
                cellLoc='center',
                loc='center'
            )

            # Adjust table style
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Hide axes
            plt.axis('off')

            # Add title
            plt.title(f'Metrics Comparison for {symbol}')

            # Save figure
            plt.tight_layout()
            plt.savefig(output_dir / "metrics_table.png", dpi=300, bbox_inches='tight')
            plt.close()

    def delete_version(self, symbol: str, version: str) -> bool:
        """
        Delete a specific model version.

        Args:
            symbol: Trading symbol or model identifier
            version: Version to delete

        Returns:
            True if successful, False otherwise
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")

        # Check if version exists
        version_dir = self.registry_path / symbol_name / version
        if not version_dir.exists():
            logger.warning(f"Version {version} not found for {symbol}")
            return False

        try:
            # Delete version directory
            shutil.rmtree(version_dir)
            logger.info(f"Deleted version {version} for {symbol}")

            # Update versions index
            index_path = self.registry_path / symbol_name / "versions_index.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    versions_index = json.load(f)

                # Remove version from index
                versions_index["versions"] = [
                    v for v in versions_index["versions"]
                    if v.get("version") != version
                ]

                # Save updated index
                with open(index_path, 'w') as f:
                    json.dump(versions_index, f, indent=2)

            return True
        except Exception as e:
            logger.error(f"Error deleting version {version} for {symbol}: {str(e)}")
            return False

    def track_online_performance(
        self,
        symbol: str,
        version: str,
        timestamp: str,
        metrics: Dict[str, float],
        prediction_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Track online performance of a model version.

        Args:
            symbol: Trading symbol or model identifier
            version: Model version
            timestamp: ISO-formatted timestamp for the performance record
            metrics: Performance metrics
            prediction_data: Optional prediction data for later analysis

        Returns:
            True if successful, False otherwise
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")

        # Check if version exists
        version_dir = self.registry_path / symbol_name / version
        if not version_dir.exists():
            logger.warning(f"Version {version} not found for {symbol}")
            return False

        try:
            # Create performance directory if it doesn't exist
            performance_dir = version_dir / "performance"
            performance_dir.mkdir(exist_ok=True)

            # Create performance record file
            record_file = performance_dir / "online_metrics.csv"

            # Initialize or append to record file
            metrics_row = {"timestamp": timestamp, **metrics}
            metrics_df = pd.DataFrame([metrics_row])

            if record_file.exists():
                # Append to existing file
                existing_df = pd.read_csv(record_file)
                updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
                updated_df.to_csv(record_file, index=False)
            else:
                # Create new file
                metrics_df.to_csv(record_file, index=False)

            # Save prediction data if provided
            if prediction_data is not None:
                predictions_file = performance_dir / "predictions.csv"
                pred_row = {"timestamp": timestamp, **prediction_data}
                pred_df = pd.DataFrame([pred_row])

                if predictions_file.exists():
                    # Append to existing file
                    existing_df = pd.read_csv(predictions_file)
                    updated_df = pd.concat([existing_df, pred_df], ignore_index=True)
                    updated_df.to_csv(predictions_file, index=False)
                else:
                    # Create new file
                    pred_df.to_csv(predictions_file, index=False)

            return True
        except Exception as e:
            logger.error(f"Error tracking performance for {symbol} version {version}: {str(e)}")
            return False

    def generate_performance_report(
        self,
        symbol: str,
        version: Optional[str] = None,
        output_path: Optional[str] = None,
        num_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a performance report for a model version.

        Args:
            symbol: Trading symbol or model identifier
            version: Model version (default: latest)
            output_path: Path to save the report (default: None)
            num_days: Number of days to include in the report (default: all)

        Returns:
            Dictionary with performance report data
        """
        # Normalize symbol name for file paths
        symbol_name = symbol.replace("/", "_")

        # Find version if not specified
        if version is None:
            versions = self.get_versions(symbol)
            if not versions:
                raise FileNotFoundError(f"No model versions found for {symbol}")
            version = versions[-1]

        # Check if version exists
        version_dir = self.registry_path / symbol_name / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version} not found for {symbol}")

        # Check if performance data exists
        performance_dir = version_dir / "performance"
        if not performance_dir.exists() or not (performance_dir / "online_metrics.csv").exists():
            logger.warning(f"No performance data found for {symbol} version {version}")
            return {"error": "No performance data found"}

        try:
            # Load performance data
            metrics_file = performance_dir / "online_metrics.csv"
            metrics_df = pd.read_csv(metrics_file)

            # Convert timestamps to datetime
            metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])

            # Filter by date if specified
            if num_days is not None:
                latest_date = metrics_df["timestamp"].max()
                cutoff_date = latest_date - pd.Timedelta(days=num_days)
                metrics_df = metrics_df[metrics_df["timestamp"] >= cutoff_date]

            # Sort by timestamp
            metrics_df = metrics_df.sort_values("timestamp")

            # Create performance report
            report = {
                "symbol": symbol,
                "version": version,
                "data_points": len(metrics_df),
                "date_range": {
                    "start": metrics_df["timestamp"].min().isoformat(),
                    "end": metrics_df["timestamp"].max().isoformat()
                }
            }

            # Calculate aggregate metrics
            numeric_cols = metrics_df.select_dtypes(include=np.number).columns
            report["metrics"] = {
                "mean": {col: metrics_df[col].mean() for col in numeric_cols},
                "std": {col: metrics_df[col].std() for col in numeric_cols},
                "min": {col: metrics_df[col].min() for col in numeric_cols},
                "max": {col: metrics_df[col].max() for col in numeric_cols},
                "latest": {col: metrics_df[col].iloc[-1] for col in numeric_cols}
            }

            # Generate visualizations if output path is provided
            if output_path is not None:
                output_dir = Path(output_path)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate metric trend plots
                for col in numeric_cols:
                    if col == "timestamp":
                        continue

                    plt.figure(figsize=(10, 6))
                    plt.plot(metrics_df["timestamp"], metrics_df[col])
                    plt.xlabel("Time")
                    plt.ylabel(col)
                    plt.title(f"{col} Over Time for {symbol} (Version: {version})")
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_dir / f"{col}_trend.png", dpi=300)
                    plt.close()

                # Save report as JSON
                report_path = output_dir / "performance_report.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)

            return report
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {"error": str(e)}


# Singleton instance of the registry
_registry_instance = None

def get_registry() -> ModelRegistry:
    """
    Get or create the singleton registry instance.

    Returns:
        The model registry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance