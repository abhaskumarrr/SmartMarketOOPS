"""
Model Evaluation Framework for Trading Models
Provides comprehensive evaluation metrics for financial and technical performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Optional
import os
import json
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation framework for trading models.
    
    Provides both financial metrics (sharpe, sortino, profit factor, max drawdown)
    and technical metrics (precision, recall, f1, etc.)
    """
    
    def __init__(
        self,
        prediction_threshold: float = 0.5,
        output_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            prediction_threshold: Threshold for binary classification
            output_dir: Directory to save evaluation results
            experiment_name: Name of the experiment
        """
        self.prediction_threshold = prediction_threshold
        
        # Set experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"evaluation_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join("evaluation", self.experiment_name)
        else:
            self.output_dir = os.path.join(output_dir, self.experiment_name)
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Metrics storage
        self.metrics = {}
        self.financial_metrics = {}
        self.technical_metrics = {}
        self.confusion_matrix = None
        self.best_threshold = prediction_threshold
    
    def evaluate_binary_classification(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        custom_threshold: Optional[float] = None,
        class_labels: Optional[List[str]] = None,
        optimize_threshold: bool = False,
        optimize_metric: str = 'f1',
    ) -> Dict[str, float]:
        """
        Evaluate binary classification model performance.
        
        Args:
            y_true: True labels (1D array)
            y_pred_proba: Predicted probabilities (1D array)
            custom_threshold: Custom threshold for classification
            class_labels: Class labels for confusion matrix
            optimize_threshold: Whether to optimize the threshold
            optimize_metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'profit_factor')
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred_proba = y_pred_proba.flatten()
        
        # Determine threshold
        threshold = custom_threshold if custom_threshold is not None else self.prediction_threshold
        
        # Optimize threshold if requested
        if optimize_threshold:
            # Find optimal threshold
            if optimize_metric == 'profit_factor':
                # Special case for financial metric
                self.best_threshold = self._find_optimal_threshold_financial(y_true, y_pred_proba)
            else:
                # For standard metrics
                self.best_threshold = self._find_optimal_threshold(y_true, y_pred_proba, optimize_metric)
            
            threshold = self.best_threshold
            
        # Apply threshold to get binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate technical metrics
        self.technical_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'threshold': threshold
        }
        
        # Calculate confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        self.technical_metrics['roc_auc'] = auc(fpr, tpr)
        
        # Calculate PR AUC (Precision-Recall AUC)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        self.technical_metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        # Calculate financial metrics
        self.financial_metrics = self._calculate_financial_metrics(y_true, y_pred)
        
        # Combine metrics
        self.metrics = {**self.technical_metrics, **self.financial_metrics}
        
        # Log metrics
        logger.info(f"Evaluation complete. Accuracy: {self.metrics['accuracy']:.4f}, F1: {self.metrics['f1_score']:.4f}")
        logger.info(f"Profit Factor: {self.metrics['profit_factor']:.4f}, Sharpe Ratio: {self.metrics['sharpe_ratio']:.4f}")
        
        # Save metrics to file
        self._save_metrics()
        
        # Create visualizations
        self.plot_confusion_matrix(class_labels=class_labels)
        self.plot_roc_curve(fpr, tpr)
        self.plot_precision_recall_curve(recall_curve, precision_curve)
        
        return self.metrics
    
    def _calculate_financial_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate financial metrics for trading models.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of financial metrics
        """
        metrics = {}
        
        # Basic trade statistics
        n_trades = (y_pred == 1).sum()
        metrics['n_trades'] = float(n_trades)
        
        # Avoid division by zero
        if n_trades == 0:
            logger.warning("No trades predicted. Financial metrics will be zero or invalid.")
            metrics.update({
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_gain': 0.0,
                'avg_loss': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
            })
            return metrics
        
        # Win rate
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        win_rate = true_positives / n_trades
        metrics['win_rate'] = float(win_rate)
        
        # Profit factor (ratio of gross profit to gross loss)
        false_positives = np.sum((y_pred == 1) & (y_true == 0))
        if false_positives > 0:
            profit_factor = true_positives / false_positives
        else:
            profit_factor = float('inf')
        metrics['profit_factor'] = float(profit_factor) if not np.isinf(profit_factor) else 10.0  # Cap at 10 if infinite
        
        # Create a "returns" series for sharpe/sortino calculation
        # For simplicity: 1 = win (profit), -1 = loss
        trade_results = np.zeros(len(y_pred))
        trade_results[(y_pred == 1) & (y_true == 1)] = 1    # Win (profit)
        trade_results[(y_pred == 1) & (y_true == 0)] = -1   # Loss
        
        # Only look at trades (where prediction is 1)
        trade_results = trade_results[y_pred == 1]
        
        if len(trade_results) > 0:
            # Sharpe ratio (mean return / std of returns) - simplified version
            mean_return = np.mean(trade_results)
            std_return = np.std(trade_results)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            metrics['sharpe_ratio'] = float(sharpe_ratio)
            
            # Sortino ratio (mean return / std of negative returns) - simplified version
            negative_returns = trade_results[trade_results < 0]
            std_negative = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            sortino_ratio = mean_return / std_negative if std_negative > 0 else 0.0
            metrics['sortino_ratio'] = float(sortino_ratio)
            
            # Calculate simulated equity curve and max drawdown
            equity_curve = np.cumsum(trade_results)
            
            # Max drawdown calculation
            max_equity = np.maximum.accumulate(equity_curve)
            drawdown = max_equity - equity_curve
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
            metrics['max_drawdown'] = float(max_drawdown)
            
            # Average gain and loss
            gains = trade_results[trade_results > 0]
            losses = trade_results[trade_results < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
            avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 0.0
            metrics['avg_gain'] = float(avg_gain)
            metrics['avg_loss'] = float(avg_loss)
            
            # Expected payoff
            expected_payoff = (win_rate * avg_gain) - ((1 - win_rate) * avg_loss)
            metrics['expected_payoff'] = float(expected_payoff)
            
            # Consecutive wins/losses
            max_consecutive_wins = self._get_max_consecutive(trade_results, 1)
            max_consecutive_losses = self._get_max_consecutive(trade_results, -1)
            metrics['max_consecutive_wins'] = int(max_consecutive_wins)
            metrics['max_consecutive_losses'] = int(max_consecutive_losses)
        else:
            # Default values if no trades
            for metric in ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'avg_gain', 'avg_loss', 'expected_payoff']:
                metrics[metric] = 0.0
            metrics['max_consecutive_wins'] = 0
            metrics['max_consecutive_losses'] = 0
        
        return metrics
    
    def _get_max_consecutive(self, values: np.ndarray, target_value: int) -> int:
        """
        Calculate maximum consecutive occurrences of a value in an array.
        
        Args:
            values: Array of values
            target_value: Value to count consecutive occurrences of
            
        Returns:
            Maximum consecutive occurrences
        """
        if len(values) == 0:
            return 0
            
        # Count consecutive occurrences
        max_consecutive = 0
        current_consecutive = 0
        
        for val in values:
            if val == target_value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> float:
        """
        Find optimal threshold for binary classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Optimal threshold
        """
        # Search for optimal threshold
        thresholds = np.linspace(0.05, 0.95, 19)  # Try thresholds from 0.05 to 0.95 in 0.05 increments
        best_metric_value = 0
        best_threshold = self.prediction_threshold
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'accuracy':
                metric_value = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                metric_value = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                metric_value = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'f1':
                metric_value = f1_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
        
        logger.info(f"Optimal threshold for {metric}: {best_threshold:.4f} (value: {best_metric_value:.4f})")
        return float(best_threshold)
    
    def _find_optimal_threshold_financial(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'profit_factor'
    ) -> float:
        """
        Find optimal threshold based on financial metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Financial metric to optimize
            
        Returns:
            Optimal threshold
        """
        # Search for optimal threshold
        thresholds = np.linspace(0.05, 0.95, 19)  # Try thresholds from 0.05 to 0.95 in 0.05 increments
        best_metric_value = 0
        best_threshold = self.prediction_threshold
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Skip if no trades
            if np.sum(y_pred) == 0:
                continue
                
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics(y_true, y_pred)
            
            # Get specific metric value
            if metric in financial_metrics:
                metric_value = financial_metrics[metric]
            else:
                # Default to profit factor
                metric_value = financial_metrics.get('profit_factor', 0)
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
        
        logger.info(f"Optimal threshold for {metric}: {best_threshold:.4f} (value: {best_metric_value:.4f})")
        return float(best_threshold)
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON file."""
        metrics_path = os.path.join(self.output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    def plot_confusion_matrix(
        self,
        class_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            class_labels: Class labels
            save_path: Path to save plot
        """
        if self.confusion_matrix is None:
            logger.warning("Confusion matrix not available")
            return
            
        if class_labels is None:
            class_labels = ['Negative', 'Positive']
            
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot confusion matrix
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            save_path: Path to save plot
        """
        if 'roc_auc' not in self.technical_metrics:
            logger.warning("ROC AUC not available")
            return
            
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.technical_metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'roc_curve.png')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"ROC curve plot saved to {save_path}")
    
    def plot_precision_recall_curve(
        self,
        recall: np.ndarray,
        precision: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            recall: Recall values
            precision: Precision values
            save_path: Path to save plot
        """
        if 'pr_auc' not in self.technical_metrics:
            logger.warning("Precision-Recall AUC not available")
            return
            
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot precision-recall curve
        plt.plot(recall, precision, label=f'PR Curve (AP = {self.technical_metrics["pr_auc"]:.4f})')
        
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        
        # Add threshold line
        if 'threshold' in self.technical_metrics:
            plt.axhline(y=self.technical_metrics['precision'], color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=self.technical_metrics['recall'], color='r', linestyle='--', alpha=0.5)
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'precision_recall_curve.png')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Precision-recall curve plot saved to {save_path}")
    
    def plot_threshold_sensitivity(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot sensitivity of metrics to threshold changes.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_path: Path to save plot
        """
        # Search thresholds
        thresholds = np.linspace(0.05, 0.95, 19)
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'profit_factor': [],
            'win_rate': [],
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate technical metrics
            metrics['accuracy'].append(accuracy_score(y_true, y_pred))
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            
            # Calculate financial metrics
            financial = self._calculate_financial_metrics(y_true, y_pred)
            metrics['profit_factor'].append(min(financial['profit_factor'], 5.0))  # Cap at 5 for plotting
            metrics['win_rate'].append(financial['win_rate'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot technical metrics
        ax1.plot(thresholds, metrics['accuracy'], label='Accuracy', marker='o')
        ax1.plot(thresholds, metrics['precision'], label='Precision', marker='s')
        ax1.plot(thresholds, metrics['recall'], label='Recall', marker='^')
        ax1.plot(thresholds, metrics['f1'], label='F1 Score', marker='x')
        
        ax1.axvline(x=self.best_threshold, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Technical Metrics vs. Threshold')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Metric Value')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot financial metrics
        ax2.plot(thresholds, metrics['profit_factor'], label='Profit Factor', marker='o')
        ax2.plot(thresholds, metrics['win_rate'], label='Win Rate', marker='s')
        
        ax2.axvline(x=self.best_threshold, color='r', linestyle='--', alpha=0.5)
        ax2.set_title('Financial Metrics vs. Threshold')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Metric Value')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'threshold_sensitivity.png')
            
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Threshold sensitivity plot saved to {save_path}")
    
    def create_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive evaluation report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Path to report file
        """
        if not self.metrics:
            logger.warning("No metrics available for report")
            return ""
            
        # Default path
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'evaluation_report.md')
        
        # Create report
        report = [
            f"# Model Evaluation Report: {self.experiment_name}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Technical Metrics",
            "",
            "| Metric | Value |",
            "| ------ | ----- |",
        ]
        
        # Add technical metrics
        for metric, value in self.technical_metrics.items():
            if isinstance(value, float):
                report.append(f"| {metric} | {value:.4f} |")
            else:
                report.append(f"| {metric} | {value} |")
        
        report.extend([
            "",
            "## Financial Metrics",
            "",
            "| Metric | Value |",
            "| ------ | ----- |",
        ])
        
        # Add financial metrics
        for metric, value in self.financial_metrics.items():
            if isinstance(value, float):
                report.append(f"| {metric} | {value:.4f} |")
            else:
                report.append(f"| {metric} | {value} |")
        
        report.extend([
            "",
            "## Confusion Matrix",
            "",
            "| | Predicted Negative | Predicted Positive |",
            "| --- | --- | --- |",
        ])
        
        # Add confusion matrix if available
        if self.confusion_matrix is not None:
            tn, fp, fn, tp = self.confusion_matrix.ravel()
            report.extend([
                f"| **Actual Negative** | {tn} | {fp} |",
                f"| **Actual Positive** | {fn} | {tp} |",
            ])
        
        # Add information about plots
        report.extend([
            "",
            "## Plots",
            "",
            "The following plots have been generated:",
            "",
            f"- Confusion Matrix: [confusion_matrix.png]({os.path.join(self.output_dir, 'confusion_matrix.png')})",
            f"- ROC Curve: [roc_curve.png]({os.path.join(self.output_dir, 'roc_curve.png')})",
            f"- Precision-Recall Curve: [precision_recall_curve.png]({os.path.join(self.output_dir, 'precision_recall_curve.png')})",
            f"- Threshold Sensitivity: [threshold_sensitivity.png]({os.path.join(self.output_dir, 'threshold_sensitivity.png')})",
        ])
        
        # Write report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Evaluation report saved to {save_path}")
        return save_path


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate true labels (60% positive, 40% negative)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # Generate predicted probabilities (correlated with true labels but with noise)
    noise = np.random.normal(0, 0.3, n_samples)
    y_pred_proba = np.clip(y_true + noise, 0, 1)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        prediction_threshold=0.5,
        experiment_name="example_evaluation"
    )
    
    # Evaluate model
    metrics = evaluator.evaluate_binary_classification(y_true, y_pred_proba, optimize_threshold=True)
    
    # Plot threshold sensitivity
    evaluator.plot_threshold_sensitivity(y_true, y_pred_proba)
    
    # Create report
    evaluator.create_evaluation_report()
    
    # Print key metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"Win Rate: {metrics['win_rate']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Optimal Threshold: {evaluator.best_threshold:.4f}") 