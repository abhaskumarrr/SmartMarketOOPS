import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, cnn_channels=64, lstm_hidden=128, lstm_layers=2, dropout=0.3, num_classes=3):
        print(f"CNNLSTMModel __init__ called with: input_size={input_size}, cnn_channels={cnn_channels}, lstm_hidden={lstm_hidden}, lstm_layers={lstm_layers}, dropout={dropout}, num_classes={num_classes}")
        super(CNNLSTMModel, self).__init__()
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes

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