import matplotlib.pyplot as plt

def plot_metrics(metrics_dict, save_path=None):
    plt.figure(figsize=(10, 6))
    for key, values in metrics_dict.items():
        plt.plot(values, label=key)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics Over Epochs')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_predictions(y_true, y_pred, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Model Predictions vs True Values')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show() 