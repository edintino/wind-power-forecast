import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from torch.utils.data import DataLoader
import settings as s
from sklearn.metrics import r2_score, root_mean_squared_error
import logging
import copy

logger = logging.getLogger(__name__)

def train_model(model, train_dataset, val_dataset, device, patience=10):
    """
    Trains a PyTorch model using train_dataset, validating after each epoch. Employs early stopping based on validation loss.

    Args:
        model (nn.Module): PyTorch model to train.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset for early stopping.
        device (torch.device): Device (CPU/GPU) to run the training on.
        patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 10.

    Returns:
        nn.Module: Model with the best validation performance.
    """

    logger.info(f"Starting training with early stopping (patience={patience})...")

    train_loader = DataLoader(train_dataset, batch_size=s.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=s.BATCH_SIZE, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=s.LR)

    best_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(s.NUM_EPOCHS):
        # --- Training ---
        model.train()
        train_losses = []
        for X_seq, y_seq in train_loader:
            X_seq, y_seq = X_seq.to(device), y_seq.to(device).unsqueeze(1)
            optimizer.zero_grad()
            output = model(X_seq)
            loss = criterion(output, y_seq)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_seq, y_seq in val_loader:
                X_seq, y_seq = X_seq.to(device), y_seq.to(device).unsqueeze(1)
                output = model(X_seq)
                loss = criterion(output, y_seq)
                val_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        logger.info(
            f"Epoch {epoch+1}/{s.NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # --- Check improvement for Early Stopping ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs of no improvement.")
                break

    # Load best model weights
    model.load_state_dict(best_model_weights)
    return model


def evaluate_forecast(preds, actuals):
    """
    Computes evaluation metrics for the given predictions and actual values.

    Args:
        preds (np.ndarray): Model predictions.
        actuals (np.ndarray): Ground truth values.

    Returns:
        tuple:
            - float: R-squared score.
            - float: Root Mean Squared Error (RMSE).
    """
    logger.info("Computing evaluation metrics...")
    r2 = r2_score(actuals, preds)
    rmse = root_mean_squared_error(actuals, preds)
    return r2, rmse