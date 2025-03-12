# Wind Power Forecasting with Temporal Fusion Transformer (TFT)

## Project Overview

This project implements an advanced wind energy forecasting solution leveraging deep learning techniques to accurately predict wind turbine active power generation. The PyTorch-based model employs the Transformer-based Temporal Fusion Transformer (TFT) architecture, optimized for robust time-series forecasting. It incorporates extensive feature engineering, standard scaling of features, weather data integration, gradient clipping for training stability, and thorough data preprocessing.

## Key Highlights
- **Model:** Temporal Fusion Transformer (TFT)
- **Framework:** PyTorch
- **Data Sources:**
  - Historical turbine data (active power, wind speed, theoretical power) [(available on Kaggle)](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)
  - Meteorological data via Meteostat
  - Astronomical calculations for day/night classification using Astral

## Technical Details and Implementation

### Data Preparation and Engineering

- **Dataset**: Historical wind turbine data recorded at 10-minute intervals.
- **Datetime Handling**: Data is sorted, indexed, and resampled to maintain completeness.
- **Feature Engineering**:
  - Determined turbine cut-in speed (minimum wind speed needed for power generation).
  - Generated temporal features: hour, month, and season.
  - Day/night classification using geographical coordinates via Astral.
  - Integrated temperature data from Meteostat API, aligned to match the data frequency.
- **Missing Data Handling**: Missing values are forward-filled, and their locations indicated via an imputation mask.
- **Derived Features**:
  - Computed delta values (changes in temperature, wind speed, wind power) to enhance trend detection.

### Data Scaling and Normalization

- **Feature Scaling**: Applied standard scaling (StandardScaler) to normalize feature distributions for improved model convergence.
- **Target Scaling**: The target variable (active power) is min-max scaled by dividing by 3600, representing the theoretical maximum output, ensuring values range between 0 and 1 for optimal model training.

### Model Architecture: Temporal Fusion Transformer (TFT)

- **Model Type**: Transformer-based model optimized specifically for accurate time-series forecasting.
- **Key Components**:
  - **Positional Encoding**: Incorporates temporal information into the model.
  - **Transformer Encoder Layers**: 3 layers with 4 attention heads each, effectively capturing complex temporal relationships.
  - **Gradient Clipping**: Implemented to prevent exploding gradients during training, enhancing training stability.
  - **Dropout Regularization**: Reduces overfitting and improves generalization.

### Data Handling for Model

- **Sliding Window Technique**: Utilizes sequences of 45 time-steps (7.5 hours) to predict one step ahead (10-minute forecast horizon).
- **PyTorch DataLoaders**: Efficient batching (batch size: 64) for training stability.
- **Train/Validation/Test Split**: Chronological split with 80% training, 10% validation, and 10% test data to realistically evaluate performance.

### Model Training and Evaluation

- **Training Procedure**:
  - Optimizer: Adam
  - Loss Function: Mean Squared Error (MSE)
  - Gradient Clipping: Applied gradient clipping to prevent exploding gradients during training.
  - Device: CUDA-enabled GPU (fallback to CPU)

- **Performance Metrics**:
  - Final Training Loss (MSE): 0.000923
  - **R² Score**: 0.983 (indicating excellent predictive capability)
  - **Root Mean Squared Error (RMSE)**: 186.055

### Interactive Visualization

- **Visualization**: Interactive Plotly graph clearly shows predicted vs. actual wind turbine active power.
- **Filtered Visualization**: Excludes data points with imputed missing values for clear evaluation.

[View Interactive Plot](forecast.html)

## Project Achievements and Highlights

- Achieved high forecasting accuracy:
  - **R² Score**: 0.985
  - **RMSE**: 186.055 (significantly better compared to XGBoost model's RMSE of ~523 using the same features)
- Successfully integrated meteorological and astronomical datasets, enhancing predictive power.
- Implemented robust missing data management, increasing model reliability in real-world scenarios.
- Introduced effective gradient clipping, mitigating potential exploding gradient issues typical in transformer models.

## Why This Matters

Accurate forecasting of wind turbine active power significantly enhances renewable energy system management, operational efficiency, and decision-making capabilities. Utilizing advanced deep learning approaches like TFT improves reliability and precision of forecasts, crucial for real-time energy management and market decisions.

## Potential Applications

- Optimized wind farm operations and grid integration
- Renewable energy production forecasting and capacity planning
- Strategic decision-making in real-time energy markets

## Further Reading

- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- [Gradient Clipping Techniques for Transformer-based Models](https://paperswithcode.com/method/gradient-clipping)

