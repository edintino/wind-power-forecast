"""
Main script to execute the entire workflow:
1) Set seeds
2) Load & preprocess data
3) Split into train & test sets
4) Build and train TFT model
5) Evaluate and visualize results
"""

import logging
import numpy as np
import settings as s
from src.data.data_preparation import load_and_preprocess_data
from src.data.dataset import TimeSeriesDataset
from src.model.tft_model import TemporalFusionTransformer
from src.train import train_model, evaluate_forecast
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Orchestrates the end-to-end process of data loading, preprocessing,
    model training, and evaluation for time-series forecasting using a
    Temporal Fusion Transformer (TFT) model.

    Steps:
        1. Set seeds for reproducibility.
        2. Load & preprocess data.
        3. Split data into train and test sets.
        4. Create dataset objects & build the TFT model.
        5. Train the model & evaluate predictions.
        6. Visualize forecasted vs actual values.
    """
    logger.info("Starting main workflow...")

    # -------------------
    # 1) Set seeds
    # -------------------
    logger.info("Setting global seeds...")
    s.set_global_seed()

    # -------------------
    # 2) Load & preprocess data
    # -------------------
    logger.info("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    # Separate target and features
    logger.info("Separating target and features...")
    y = df["active_power"].astype(np.float32).values / 3600
    X = df.drop(columns=["active_power"]).astype(np.float32).values

    # -------------------
    # 3) Train-Val-Test split
    # -------------------
    total_len = len(X)
    train_end_idx = int(total_len * s.TRAIN_RATIO)
    val_end_idx = train_end_idx + int(total_len * s.VAL_RATIO)

    X_train, X_val, X_test = X[:train_end_idx], X[train_end_idx:val_end_idx], X[val_end_idx:]
    y_train, y_val, y_test = y[:train_end_idx], y[train_end_idx:val_end_idx], y[val_end_idx:]

    # Scale only on train, then transform val and test
    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create dataset objects
    train_dataset = TimeSeriesDataset(X_train, y_train, s.SEQ_LENGTH, s.FORECAST_HORIZON)
    val_dataset = TimeSeriesDataset(X_val, y_val, s.SEQ_LENGTH, s.FORECAST_HORIZON)
    test_dataset = TimeSeriesDataset(X_test, y_test, s.SEQ_LENGTH, s.FORECAST_HORIZON)

    # -------------------
    # 4) Datasets & Model
    # -------------------
    logger.info("Creating TimeSeriesDataset objects...")
    train_dataset = TimeSeriesDataset(X_train, y_train, s.SEQ_LENGTH, s.FORECAST_HORIZON)
    test_dataset = TimeSeriesDataset(X_test, y_test, s.SEQ_LENGTH, s.FORECAST_HORIZON)

    logger.info("Initializing Temporal Fusion Transformer model...")
    model = TemporalFusionTransformer(
        input_size=X_train.shape[1] + 1,  # +1 for the appended historical target
        hidden_size=s.HIDDEN_SIZE,
        output_size=s.OUTPUT_SIZE,
        num_heads=s.NUM_HEADS,
        num_layers=s.NUM_LAYERS,
        dropout=s.DROPOUT
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}", )
    model.to(device)

    # -------------------
    # 5) Train & Evaluate
    # -------------------
    logger.info("Training the model...")
    model = train_model(
        model,
        train_dataset,
        val_dataset,
        device,
        patience=s.PATIENCE
    )

    # -------------------
    # Evaluate on Test Set
    # -------------------
    logger.info("Evaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=s.BATCH_SIZE, shuffle=False)

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_seq, y_seq in test_loader:
            X_seq, y_seq = X_seq.to(device), y_seq.to(device).unsqueeze(1)
            output = model(X_seq)
            preds.append(output.cpu().numpy())
            actuals.append(y_seq.cpu().numpy())

    preds = 3600 * np.concatenate(preds).squeeze()
    actuals = 3600 * np.concatenate(actuals).squeeze()

    r2, rmse = evaluate_forecast(preds, actuals)
    print(f"Final Test R2 Score: {r2:.3f}")
    print(f"Final Test RMSE: {rmse:.3f}")

    # -------------------
    # 6) Visualization
    # -------------------
    logger.info("Generating forecast visualization...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(actuals))),
        y=actuals,
        mode='lines+markers',
        name='Actual'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(preds))),
        y=preds,
        mode='lines+markers',
        name='Forecast'
    ))
    fig.update_layout(
        title='TFT: Actual vs. Forecasted Active Power',
        xaxis_title='Test Sample Index',
        yaxis_title='Active Power'
    )
    fig.write_html("forecast.html")
    logger.info("Workflow completed. Forecast saved as forecast.html")

if __name__ == "__main__":
    main()