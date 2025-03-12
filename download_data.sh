#!/bin/bash
# Create destination folder
mkdir -p data

# Download dataset using curl and save as a ZIP file in the destination folder
curl -L -o data/wind-turbine-scada-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/berkerisen/wind-turbine-scada-dataset

# Unzip the downloaded file into the destination folder
unzip -o data/wind-turbine-scada-dataset.zip -d data

# (Optional) Remove the ZIP file after extraction
rm ./data/wind-turbine-scada-dataset.zip

echo "Data downloaded and extracted to ./data/"
