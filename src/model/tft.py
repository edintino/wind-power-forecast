import math
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TFTModel(nn.Module):
    """
    Enhanced Temporal Fusion Transformer for sequence forecasting.
    """
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        logger.debug(f"Initializing Enhanced TFT with input_size={input_size}, hidden_size={hidden_size}")

        self.input_projection = nn.Linear(input_size, hidden_size)
        # Option to switch between learnable and sinusoidal positional encoding
        self.pos_encoder = LearnablePositionalEncoding(hidden_size)  
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,  # Increased feedforward dimension
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        x = self.input_projection(src)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Residual connection: add the last input feature directly
        residual = self.input_projection(src[:, -1, :])
        x = self.layer_norm(x[:, -1, :] + residual)
        return self.output_layer(x)