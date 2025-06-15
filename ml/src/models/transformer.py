import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.3, num_classes=3):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.num_classes = num_classes

        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)

        # Permute x for TransformerEncoder: (seq_len, batch, input_size)
        x = x.permute(1, 0, 2)

        # Embedding layer
        x = self.embedding(x) * math.sqrt(self.d_model)
        # x shape: (seq_len, batch, d_model)

        # Positional Encoding
        x = self.pos_encoder(x)
        # x shape: (seq_len, batch, d_model)

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x)
        # transformer_output shape: (seq_len, batch, d_model)

        # Take the output from the last time step for classification
        # transformer_output[-1, :, :] shape: (batch, d_model)

        # Pass through dropout
        dropped_out = self.dropout(transformer_output[-1, :, :])

        # Pass through the fully connected layer
        output = self.fc(dropped_out)
        # output shape: (batch, num_classes)

        return output 