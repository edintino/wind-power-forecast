import math
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Applies positional encoding to input embeddings to preserve positional
    information in sequence data.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initializes the positional encoding module.

        Args:
            d_model (int): Dimensionality of the model.
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.1.
            max_len (int, optional): Maximum length of positional embeddings. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].

        Returns:
            torch.Tensor: Tensor with positional encoding applied.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class TemporalFusionTransformer(nn.Module):
    """
    A simplified Temporal Fusion Transformer for sequence forecasting.
    """
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers, dropout=0.1):
        """
        Initializes the TFT model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Dimensionality of hidden embeddings.
            output_size (int): Dimension of the output (forecast horizon).
            num_heads (int): Number of heads in the multi-head attention.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float, optional): Dropout rate for Transformer layers. Defaults to 0.1.
        """
        super().__init__()
        logger.debug(f"Initializing TemporalFusionTransformer with input_size={input_size}, hidden_size={hidden_size}")

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        """
        Defines the forward pass of the Temporal Fusion Transformer model.

        Args:
            src (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_size].

        Returns:
            torch.Tensor: Forecasted values of shape [batch_size, output_size].
        """
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.output_layer(x[:, -1, :])