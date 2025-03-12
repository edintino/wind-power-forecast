# main.py

"""
Main script to execute the entire workflow:
1) Set seeds
2) Load & preprocess data
3) Split into train, validation, and test sets
4) Build and train the chosen model (TFT, LSTM)
5) Evaluate and visualize results for all models
"""

import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import settings as s
from src.data.data_preparation import load_and_preprocess_data
from src.data.dataset import TimeSeriesDataset
from src.model.tft import TFTModel
from src.model.lstm import LSTMModel
from src.train import train_model, evaluate_forecast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="TFT",
        choices=["TFT", "LSTM"],
        help="Which model to run (TFT, LSTM). Default is 'TFT'."
    )
    return parser.parse_args()

def test_and_plot(model, test_dataset, device, model_name):
    """
    Runs inference on the test set, creates an HTML plot of the
    actual vs. predicted values, and returns both arrays.
    """
    test_loader = DataLoader(test_dataset, batch_size=s.BATCH_SIZE, shuffle=False)
    model.eval()
    preds_list, actuals_list = [], []

    with torch.no_grad():
        for X_seq, y_seq in test_loader:
            X_seq, y_seq = X_seq.to(device), y_seq.to(device).unsqueeze(1)
            outputs = model(X_seq)  # shape [batch_size, 1]
            preds_list.append(outputs.cpu().numpy())
            actuals_list.append(y_seq.cpu().numpy())

    preds = 3600 * np.concatenate(preds_list).squeeze()   # revert /3600
    actuals = 3600 * np.concatenate(actuals_list).squeeze()

    # Plot using Plotly
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
        title=f'{model_name} - Actual vs Forecasted Active Power',
        xaxis_title='Test Sample Index',
        yaxis_title='Active Power'
    )
    # Save with model prefix
    output_file = f"{model_name.lower()}_forecast.html"
    fig.write_html(output_file)
    logger.info(f"Saved forecast visualization: {output_file}")

    return preds, actuals

def main():
    # 1) Parse arguments & set seeds
    logger.info("Parsing command-line arguments...")
    args = parse_arguments()
    model_name = args.model.upper()

    logger.info("Setting global seeds...")
    s.set_global_seed()

    # 2) Load & preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    # Separate target and features
    y = df["active_power"].astype(np.float32).values / 3600
    X = df.drop(columns=["active_power"]).astype(np.float32).values

    # 3) Train-Val-Test split
    total_len = len(X)
    train_end_idx = int(total_len * s.TRAIN_RATIO)
    val_end_idx = train_end_idx + int(total_len * s.VAL_RATIO)

    X_train, X_val, X_test = X[:train_end_idx], X[train_end_idx:val_end_idx], X[val_end_idx:]
    y_train, y_val, y_test = y[:train_end_idx], y[train_end_idx:val_end_idx], y[val_end_idx:]

    # 4) Scale features
    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create Datasets
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train, s.SEQ_LENGTH, s.FORECAST_HORIZON)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val, s.SEQ_LENGTH, s.FORECAST_HORIZON)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test, s.SEQ_LENGTH, s.FORECAST_HORIZON)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5) Initialize the selected model
    if model_name == "TFT":
        model = TFTModel(
            input_size=X_train.shape[1] + 1,
            hidden_size=s.HIDDEN_SIZE,
            output_size=s.OUTPUT_SIZE,
            num_heads=s.NUM_HEADS,
            num_layers=s.NUM_LAYERS,
            dropout=s.DROPOUT
        ).to(device)

    else:  # LSTM
        model = LSTMModel(
            input_size=X_train.shape[1] + 1,
            hidden_size=s.HIDDEN_SIZE,
            num_layers=s.NUM_LAYERS,
            dropout=s.DROPOUT
        ).to(device)

    # Train & Test
    logger.info(f"Training the {model_name} model with early stopping...")
    model = train_model(model, train_dataset, val_dataset, device, patience=s.PATIENCE)
    logger.info(f"Evaluating {model_name} on the test set...")
    preds, actuals = test_and_plot(model, test_dataset, device, model_name)

    # 6) Compute final metrics
    logger.info("Computing evaluation metrics (R2, RMSE)...")
    r2, rmse = evaluate_forecast(preds, actuals)
    logger.info(f"{model_name} -> R2: {r2:.3f}, RMSE: {rmse:.3f}")
    print(f"Final Test R² ({model_name}): {r2:.3f}")
    print(f"Final Test RMSE ({model_name}): {rmse:.3f}")

if __name__ == "__main__":
    main()
