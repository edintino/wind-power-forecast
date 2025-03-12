"""
Executes the entire wind power forecasting workflow:

1. Parses command-line arguments and sets random seeds for reproducibility.
2. Loads and preprocesses wind turbine data, integrating weather information.
3. Splits dataset chronologically into train, validation, and test subsets.
4. Scales features using StandardScaler.
5. Builds, trains, and evaluates the specified forecasting model (TFT, LSTM, or XGBoost).
6. Visualizes actual versus predicted power generation for performance comparison.
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
from src.model.xgboost import XGBoostModel
from src.train import train_model, evaluate_forecast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="XGBOOST",
        choices=["TFT", "LSTM", "XGBOOST"],
        help="Which model to run (TFT, LSTM or XGBOOST). Default is 'XGBOOST'."
    )
    return parser.parse_args()

def plot_forecast(actuals, preds, model_name):
    """
    Creates and saves an interactive Plotly visualization comparing actual vs. predicted active power.

    Args:
        actuals (np.ndarray): Ground truth active power values.
        preds (np.ndarray): Forecasted active power values.
        model_name (str): Name of the model (TFT, LSTM, XGBoost).
    """
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
        title=f"{model_name} - Actual vs Forecast",
        xaxis_title='Test Sample Index',
        yaxis_title='Active Power'
    )
    fig.write_html(f"{model_name.lower()}_forecast.html")
    logger.info(f"Forecast visualization saved: {model_name.lower()}_forecast.html")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, actuals = None, None

    # 5) Initialize & train the selected model
    if model_name in ["TFT", "LSTM"]:
        # Create PyTorch Datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train, s.SEQ_LENGTH, s.FORECAST_HORIZON)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val, s.SEQ_LENGTH, s.FORECAST_HORIZON)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test, s.SEQ_LENGTH, s.FORECAST_HORIZON)

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

        model = train_model(model, train_dataset, val_dataset, device, patience=s.PATIENCE)
        # Collect test predictions
        test_loader = DataLoader(test_dataset, batch_size=s.BATCH_SIZE, shuffle=False)
        preds_list, actuals_list = [], []
        model.eval()
        with torch.no_grad():
            for X_seq, y_seq in test_loader:
                X_seq, y_seq = X_seq.to(device), y_seq.to(device).unsqueeze(1)
                outputs = model(X_seq)
                preds_list.append(outputs.cpu().numpy())
                actuals_list.append(y_seq.cpu().numpy())
        preds = 3600 * np.concatenate(preds_list).squeeze()
        actuals = 3600 * np.concatenate(actuals_list).squeeze()

    elif model_name == "XGBOOST":
        # Flatten data approach for single-step forecast
        xgb_model = XGBoostModel(n_estimators=300, learning_rate=0.03, max_depth=6)
        xgb_model.fit(X_train_scaled, y_train, X_val_scaled, y_val)
        fc = xgb_model.predict(X_test_scaled)
        preds = fc * 3600
        actuals = y_test * 3600

    # 6) Evaluate & Plot
    r2, rmse = evaluate_forecast(preds, actuals)
    logger.info(f"{model_name} -> R2: {r2:.3f}, RMSE: {rmse:.3f}")
    print(f"Final Test R² ({model_name}): {r2:.3f}")
    print(f"Final Test RMSE ({model_name}): {rmse:.3f}")

    plot_forecast(actuals, preds, model_name)

if __name__ == "__main__":
    main()