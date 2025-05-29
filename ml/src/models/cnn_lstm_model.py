"""
CNN-LSTM Hybrid Model Architecture for cryptocurrency trading with Smart Money Concepts features
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F

# Commented out to resolve circular import
# from .base_model import BaseModel

logger = logging.getLogger(__name__)


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for time-series prediction of cryptocurrency markets
    with Smart Money Concepts features.

    This model combines CNN layers for feature extraction with LSTM layers
    for temporal modeling. The CNN effectively captures local patterns while
    the LSTM handles temporal dependencies.
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        device: str | None = None,
        **kwargs,
    ):
        """
        Initialize the CNN-LSTM model.

        Args:
            input_size: Dimension of input features
            cnn_channels: Number of channels in CNN layers
            lstm_hidden: Number of units in LSTM layers
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            num_classes: Number of output classes
            device: Device to use for computation
            **kwargs: Additional arguments for model initialization
        """
        print(f"CNNLSTMModel __init__ called with: input_size={input_size}, cnn_channels={cnn_channels}, lstm_hidden={lstm_hidden}, lstm_layers={lstm_layers}, dropout={dropout}, num_classes={num_classes}")
        super(CNNLSTMModel, self).__init__()
        self.input_size = input_size
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # LSTM layers (input to LSTM will be cnn_channels*2 * pooled_sequence_length)
        # Need to calculate the output size of CNN first to determine LSTM input size
        # Let's assume input sequence length is L
        # After first conv/pool: L/2
        # After second conv/pool: L/4
        # So LSTM input size is cnn_channels*2 * (seq_len // 4)
        # We'll adjust this dynamically or pass it.
        # For now, let's assume seq_len is a multiple of 4 and use a placeholder
        # A better approach would be to use a dummy tensor to calculate this dynamically.
        # Example: if seq_len=60, pooled_seq_len = (60 // 2) // 2 = 15
        # LSTM input size = cnn_channels*2 * 15
        
        # A more robust way to get LSTM input size after CNN. Pass a dummy tensor:
        # dummy_input = torch.randn(1, input_size, 60) # (batch, features, seq_len)
        # cnn_output = self.cnn(dummy_input)
        # lstm_input_size = cnn_output.view(cnn_output.size(0), cnn_output.size(1), -1).shape[2]
        # This dynamic calculation is better in a setup function or forward pass initial check.
        # For now, let's use a common pattern: permute CNN output for LSTM input (seq_len, batch, features)
        # LSTM expects input shape (seq_len, batch, input_size)
        # CNN output shape is (batch, channels, seq_len_after_pool)
        # We need to permute to (seq_len_after_pool, batch, channels)

        # We will use the actual pooled sequence length derived from forward pass.

        self.lstm = nn.LSTM(input_size=cnn_channels*2, # input size to LSTM after CNN pooling
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=dropout,
                            batch_first=True) # Use batch_first for easier handling

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer (Output layer for classification)
        # The input to the linear layer is the output of the last LSTM cell
        self.fc = nn.Linear(lstm_hidden, num_classes) # Changed output size to num_classes
        self.to(self.device)

        if kwargs.get("model_path") and os.path.exists(kwargs["model_path"]):
            try:
                self.load_state_dict(torch.load(kwargs["model_path"], map_location=self.device))
                logger.info(f"Loaded weights from {kwargs['model_path']}")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # Permute x for CNN: (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)

        # Pass through CNN
        cnn_out = self.cnn(x)
        # cnn_out shape: (batch, cnn_channels*2, seq_len_after_pool)

        # Permute cnn_out for LSTM: (batch, seq_len_after_pool, cnn_channels*2)
        lstm_in = cnn_out.permute(0, 2, 1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(lstm_in)
        # lstm_out shape: (batch, seq_len_after_pool, lstm_hidden)

        # Get the output from the last time step
        # lstm_out[:, -1, :] shape: (batch, lstm_hidden)

        # Pass through dropout
        dropped_out = self.dropout(lstm_out[:, -1, :])

        # Pass through the fully connected layer
        output = self.fc(dropped_out)
        # output shape: (batch, num_classes) - Raw scores before Softmax

        # Note: Softmax is typically applied with the CrossEntropyLoss in PyTorch,
        # so we don't apply it here in the forward pass unless specified.

        return output

    def fit_model(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        lr: float = 0.001,
        early_stopping_patience: int = 10,
        checkpoint_dir: str | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Train the CNN-LSTM model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of training epochs
            lr: Learning rate for optimizer
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory for model checkpoints
            class_weights: Optional tensor of class weights for the loss function

        Returns:
            Dictionary with training history
        """
        if self.num_classes == 1:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=class_weights.to(self.device) if class_weights is not None else None
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            for x, y in train_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float() if self.num_classes == 1 else y.to(self.device).long()
                optimizer.zero_grad()
                outputs = self(x)
                if self.num_classes == 1:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y)
                else:
                    loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_loss = self._evaluate_loss(val_loader, criterion)
            scheduler.step(val_loss)
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {np.mean(train_losses):.4f} - Val Loss: {val_loss:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.state_dict()
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(self.state_dict(), os.path.join(checkpoint_dir, "best_cnnlstm.pt"))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break

        if best_state:
            self.load_state_dict(best_state)
        return {"best_val_loss": best_val_loss}

    def _evaluate_loss(self, loader, criterion):
        self.eval()
        losses = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device).float()
                y = y.to(self.device).float() if self.num_classes == 1 else y.to(self.device).long()
                outputs = self(x)
                if self.num_classes == 1:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y)
                else:
                    loss = criterion(outputs, y)
                losses.append(loss.item())
        return np.mean(losses)

    def predict(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            x: Input data
            batch_size: Batch size for prediction

        Returns:
            Predictions
        """
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = torch.tensor(x[i : i + batch_size]).to(self.device).float()
                out = self(batch)
                if self.num_classes == 1:
                    out = out.cpu().numpy().flatten()
                else:
                    out = torch.softmax(out, dim=1).cpu().numpy()
                preds.append(out)
        if self.num_classes == 1:
            return np.concatenate(preds)
        else:
            return np.vstack(preds)

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 64) -> dict:
        """
        Evaluate the model performance.

        Args:
            x: Input data
            y: True labels
            batch_size: Batch size for prediction

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(x, batch_size)
        if self.num_classes == 1:
            y_pred_label = (y_pred > 0.5).astype(int)
            y_true = y.astype(int).flatten()
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred_label),
                "precision": precision_score(y_true, y_pred_label),
                "recall": recall_score(y_true, y_pred_label),
                "f1_score": f1_score(y_true, y_pred_label),
            }
        else:
            y_pred_label = np.argmax(y_pred, axis=1)
            y_true = y.astype(int).flatten()
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred_label),
                "precision": precision_score(y_true, y_pred_label, average="macro"),
                "recall": recall_score(y_true, y_pred_label, average="macro"),
                "f1_score": f1_score(y_true, y_pred_label, average="macro"),
            }
        return metrics

    def save(self, model_path: str) -> None:
        """
        Save the model.

        Args:
            model_path: Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str, **kwargs) -> "CNNLSTMModel":
        """
        Load a pre-trained model.

        Args:
            model_path: Path to the saved model
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model
        """
        model = cls(**kwargs)
        model.load_state_dict(torch.load(model_path, map_location=kwargs.get("device", "cpu")))
        model.eval()
        return model

    def visualize_feature_maps(
        self, x: np.ndarray, layer_names: list[str] | None = None
    ) -> dict[str, np.ndarray]:
        """
        Visualize feature maps from CNN layers.

        Args:
            x: Input data
            layer_names: Names of layers to visualize

        Returns:
            Dictionary mapping layer names to feature maps
        """
        if layer_names is None:
            # Get all convolutional layer names
            layer_names = [layer.name for layer in self.modules() if isinstance(layer, nn.Conv1d)]

        # Create models to output feature maps
        feature_maps = {}
        for layer_name in layer_names:
            layer = next(
                (
                    layer_obj
                    for layer_obj in self.modules()
                    if layer_obj.__class__.__name__ == layer_name
                ),
                None,
            )
            if layer is not None:
                intermediate_model = nn.Sequential(layer, nn.Flatten())
                # Get feature maps for the first sample
                feature_maps[layer_name] = (
                    intermediate_model(torch.tensor(x[:1]).permute(0, 2, 1)).cpu().numpy()
                )

        return feature_maps

    def get_feature_importance(self, x: np.ndarray, n_samples: int = 10) -> dict[str, float]:
        """
        Get feature importance using permutation importance.

        Args:
            x: Input data
            n_samples: Number of samples to use

        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if len(x) < n_samples:
            n_samples = len(x)

        # Select random samples
        indices = np.random.choice(len(x), n_samples, replace=False)
        x_samples = x[indices]

        # Get baseline predictions
        baseline_pred = self.predict(x_samples)

        # Calculate importance for each feature
        importance_scores = {}

        for feature_idx in range(x_samples.shape[2]):
            # Create permuted data
            x_permuted = x_samples.copy()

            # Permute the feature across all time steps
            for i in range(x_samples.shape[0]):
                perm_indices = np.random.permutation(x_samples.shape[1])
                x_permuted[i, :, feature_idx] = x_samples[i, perm_indices, feature_idx]

            # Get predictions with permuted feature
            permuted_pred = self.predict(x_permuted)

            # Calculate importance as mean absolute difference
            importance = np.mean(np.abs(baseline_pred - permuted_pred))
            importance_scores[feature_idx] = float(importance)

        # Sort by importance (descending)
        importance_scores = dict(
            sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_scores

    def _build_model(self):
        pass
