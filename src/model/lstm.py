import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    A simple LSTM model for time-series forecasting, returning a single-step prediction.
    Uses the same input shape as your TFT: (batch_size, seq_length, num_features + 1).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass for LSTM.
        Args:
            x (torch.Tensor): Shape [batch_size, seq_length, input_size].
        Returns:
            torch.Tensor: Shape [batch_size, 1] (the predicted value).
        """
        out, (_, _) = self.lstm(x)
        # We take the last time step's output to pass into the final FC
        last_out = out[:, -1, :]
        return self.fc(last_out)
