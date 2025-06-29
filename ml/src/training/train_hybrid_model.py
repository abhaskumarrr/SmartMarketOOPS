import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ml.src.features.build_features import build_features
from ml.src.features.generate_signals import generate_signals

# A more complex model to handle the new features
class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)  # 0: hold, 1: buy, 2: sell
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("--- Starting Hybrid Model Training ---")

    # 1. Load and Process Data
    data_path = 'data/raw/BTC_USDT_1h.csv'
    df = pd.read_csv(data_path)
    
    features_df = build_features(df)
    signals_df = generate_signals(df, features_df)
    
    # For this example, we'll train on the 'trend_capture' strategy signals
    labels = signals_df['trend_capture']
    
    # Align data
    aligned_df = features_df.join(labels, how='inner')
    
    X = aligned_df.drop(columns=['trend_capture', 'regime', 'bias']).values
    y = aligned_df['trend_capture'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 2. Initialize Model, Loss, and Optimizer
    input_dim = X_train.shape[1]
    model = HybridModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 30
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}%")

    # 4. Save the Model
    model_path = 'ml/models/hybrid_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"--- Hybrid model saved to {model_path} ---")

if __name__ == "__main__":
    main()
