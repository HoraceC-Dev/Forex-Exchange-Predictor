<div align="center">
    <h1>Forex Exchange Predictor with LSTM</h1>
</div>

# Overview
This project aims to predict the trends in the GBP/USD forex pair on a 4-hour timeframe using Short-Term Memory (LSTM) neural networks, a type of Recurrent Neural Network (RNN). The LSTM model is trained using 5000 data points, each representing GBP/USD movements over 4-hour intervals, with attributes including open, close, high, low, volume, and a corresponding trend label. By utilizing a predefined period of historical data as input, the model effectively captures temporal patterns and dependencies in the time-series data, enabling it to predict the trend labels for n future time steps.

# Dataset
**Features**:
- Open price
- Close price
- Highest price
- Lowest price
- Volume
- Swing High(In code calculation)
- Swing Low(In code calculation)
- EMA_150 (In code calculation)
- EMA_200 (In code calculation)

**Target(Label)**:
- Class 1: -1 (Down Trend)
- Class 2: 0 (Netural)
- Class 3: 1 (Up Trend)

## Sequence Format & Variable
Tensor Shape: [# of data points, look_back_window, input_features]

**look_back_window**: The number of past time step
**input_features**: The number of features (In this project it willl 9)

# Model Architecture
The LSTM model is designed to capture the sequential nature of the forex data. Key layers and parameters include:

`Input Layer`: Processes a sequence of historical features (look-back window).

`LSTM Layers`: Two stacked LSTM layers to capture temporal dependencies.

`Dropout Layer`: To reduce overfitting.

`Fully Connected Layer`: Maps LSTM output to trend predictions.

`Activation Function`: Softmax for multi-class trend prediction.

Hyperparameters used during training:
| # | Hyperparameters        | Value     |
|-----------------|----------------|----------------|
| 1| Look-back Window| 10|
| 2| Learning rate   | 0.001|
| 3| Optimizer  | Adam|
| 4| Loss function   | CrossEntropyLoss|
| 5| Batch size  | 64|
| 6| Epochs   | 200|

# Installation
Clone the repository:
```bash
git clone https://github.com/HoraceC-Dev/Forex-Exchange-Predictor.git
cd Forex-Exchange-Predictor
```
Install dependencies:
```bash
pip install -r requirements.txt
```
# Usage
### 1. Dataset and Technical Analysis

- The `dataset/` folder contains:
  - gbpusd_data.csv: The dataset used for training, with 5000 data points and attributes such as open, close, high, low, volume, and trend labels.
- The `graphs/` folder contains:
  - Technical Analysis Graphs: Visualizations that provide insights into the dataset and the model's performance. 

### 2. Training the Model
For a comprehensive guide on the training process, refer to the provided Jupyter Notebook:

Notebook: `model_training.ipynb`
- Walks through the entire training pipeline, from preprocessing to model evaluation.
    - Includes parameters that can be adjusted to fine-tune the LSTM model.Allows users to experiment with different configurations like:
        - Look-back window size
        - Batch size
        - Learning rate
        - Number of epochs
### 3. Pretrained Model
The repository includes a pretrained LSTM model for immediate use:

Path: `models/model.pth`

You can load this model for evaluation or further fine-tuning without retraining from scratch.

# License
This project is licensed under the MIT License - see the [LICENSE](License) file for details.

# Disclaimer
The Forex-Exchange-Predictor is intended for educational purposes only. The creator assumes no responsibility for any consequences arising from it use. Users are advised to comply with appropriate terms of service of relevant usage and adhere to all applicable laws, regulartions, and ethical guidlines.