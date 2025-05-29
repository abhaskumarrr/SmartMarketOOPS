import optuna
import torch
from torch.utils.data import DataLoader
from .trainer import Trainer

def objective(trial, model_class, dataset, val_dataset, device, class_weights=None):
    try:
        cnn_channels = trial.suggest_int('cnn_channels', 8, 64)
        lstm_hidden = trial.suggest_int('lstm_hidden', 32, 256)
        lstm_layers = trial.suggest_int('lstm_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        input_size = dataset[0][0].shape[-1]  # dataset[0] is (window, target)
        model = model_class(input_size=input_size, cnn_channels=cnn_channels, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout, num_classes=3).to(device)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True))

        for epoch in range(30):
            model.train()
            total_loss = 0
            for batch_idx, (features, targets) in enumerate(train_loader):
                features, targets = features.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(features.permute(0, 2, 1))
                
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}:")
                print(f"  Model Output Shape: {outputs.shape}")
                print(f"  Target Shape: {targets.shape}")
                if class_weights is not None:
                    print(f"  Class Weights: {class_weights.to(device)}")
                loss = criterion(outputs, targets)
                print(f"  Batch Loss: {loss.item()}")

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Avg Train Loss: {avg_train_loss}")

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    outputs = model(features.permute(0, 2, 1))
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} Avg Val Loss: {avg_val_loss}")

            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_val_loss

    except Exception as e:
        print(f"[Optuna Trial Exception] {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

def run_optuna(model_class, dataset, val_dataset, device, n_trials=20, timeout=None, pruner=None, class_weights=None):
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_study.db",
        heartbeat_interval=60,
        grace_period=120
    )
    study = optuna.create_study(
        direction='minimize',
        storage=storage,
        load_if_exists=True,
        pruner=pruner
    )
    study.optimize(
        lambda trial: objective(trial, model_class, dataset, val_dataset, device, class_weights),
        n_trials=n_trials,
        timeout=timeout,
        catch=(Exception,)
    )
    print('Best trial:', study.best_trial.params)
    return study.best_trial 