"""
Transformer Model Architecture for cryptocurrency trading with Smart Money Concepts features
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .base_model import BaseModel
from typing import Optional

class TransformerModel(BaseModel, nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, forecast_horizon, d_model=64, nhead=4, num_layers=2, dropout=0.2, device=None, **kwargs):
        BaseModel.__init__(self)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.Sigmoid() if output_dim == 1 else nn.Identity()
        )
        self.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        if x.dim() == 3:
            x = x[:, -1, :]
        out = self.fc(x)
        return out

    def fit_model(self, train_loader, val_loader, num_epochs=100, lr=0.001, early_stopping_patience=10, checkpoint_dir=None, class_weights: Optional[torch.Tensor] = None):
        if self.output_dim == 1:
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.device) if class_weights is not None else None)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            for X, y in train_loader:
                X = X.to(self.device).float()
                y = y.to(self.device).float() if self.output_dim == 1 else y.to(self.device).long()
                optimizer.zero_grad()
                outputs = self(X)
                if self.output_dim == 1:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y)
                else:
                    loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            val_loss = self._evaluate_loss(val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.state_dict()
                if checkpoint_dir:
                    import os
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(self.state_dict(), os.path.join(checkpoint_dir, "best_transformer.pt"))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        if best_state:
            self.load_state_dict(best_state)
        return {"best_val_loss": best_val_loss}

    def _evaluate_loss(self, loader, criterion):
        self.eval()
        losses = []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device).float()
                y = y.to(self.device).float() if self.output_dim == 1 else y.to(self.device).long()
                outputs = self(X)
                if self.output_dim == 1:
                    outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, y)
                else:
                    loss = criterion(outputs, y)
                losses.append(loss.item())
        return np.mean(losses)

    def predict(self, X: np.ndarray, batch_size=64) -> np.ndarray:
        self.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i+batch_size]).to(self.device).float()
                out = self(batch)
                if self.output_dim == 1:
                    out = out.cpu().numpy().flatten()
                else:
                    out = torch.softmax(out, dim=1).cpu().numpy()
                preds.append(out)
        if self.output_dim == 1:
            return np.concatenate(preds)
        else:
            return np.vstack(preds)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size=64) -> dict:
        y_pred = self.predict(X, batch_size)
        if self.output_dim == 1:
            y_pred_label = (y_pred > 0.5).astype(int)
            y_true = y.astype(int).flatten()
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred_label),
                'precision': precision_score(y_true, y_pred_label),
                'recall': recall_score(y_true, y_pred_label),
                'f1_score': f1_score(y_true, y_pred_label)
            }
        else:
            y_pred_label = np.argmax(y_pred, axis=1)
            y_true = y.astype(int).flatten()
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred_label),
                'precision': precision_score(y_true, y_pred_label, average='macro'),
                'recall': recall_score(y_true, y_pred_label, average='macro'),
                'f1_score': f1_score(y_true, y_pred_label, average='macro')
            }
        return metrics
    
    def save(self, model_path: str) -> None:
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.state_dict(), model_path)
    
    @classmethod
    def load(cls, model_path: str, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(model_path, map_location=kwargs.get('device', 'cpu')))
        model.eval()
        return model
    
    def _build_model(self):
        pass 