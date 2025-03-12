import torch
import numpy as np
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """
    Creates sliding windows from features and targets for time-series forecasting.
    """
    def __init__(self, features, targets, seq_length, forecast_horizon):
        """
        Args:
            features (np.ndarray): Feature data of shape [num_samples, num_features].
            targets (np.ndarray): Target data of shape [num_samples].
            seq_length (int): Length of the historical window for each sample.
            forecast_horizon (int): Offset of the forecast step.
        """
        super().__init__()
        self.X = features
        self.y = targets
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        logger.debug(
            f"TimeSeriesDataset initialized with {len(self.X)} samples, seq_length={seq_length}, forecast_horizon={forecast_horizon}",
        )

    def __len__(self):
        """
        Returns:
            int: Number of valid sequences in the dataset.
        """
        return len(self.X) - self.seq_length - self.forecast_horizon + 1

    def __getitem__(self, idx):
        """
        Retrieves a sequence of features and the corresponding forecast target.

        Args:
            idx (int): Index into the dataset.

        Returns:
            tuple:
                - torch.Tensor: Sequence of features with shape [seq_length, num_features + 1].
                - torch.Tensor: Single forecast target value.
        """
        feature_seq = self.X[idx : idx + self.seq_length]
        target_seq = self.y[idx : idx + self.seq_length].reshape(-1, 1)

        # Combine current features with corresponding historical target
        x_seq = np.hstack((feature_seq, target_seq))

        # Forecast target is the value at (seq_length + forecast_horizon - 1)
        forecast_target = self.y[idx + self.seq_length + self.forecast_horizon - 1]

        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(forecast_target, dtype=torch.float32)