import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

def calculate_ema(prices, period):
    prices_series = pd.Series(prices)
    ema = prices_series.ewm(span=period, adjust=False).mean()
    return ema

# Load and preprocess data
def get_data(path):
    temp_df = pd.read_csv(path)
    df = temp_df['Time\tOpen\tHigh\tLow\tClose\tVolume'].str.split('\t', expand=True)
    df["Label"] = temp_df["Label"] +1

    df = df.drop(columns=[6])
    df.columns = ['Time', 'Open', 'High', 'Low', "Close", "Volume" , "Label"]
    df = df.drop(columns=["Time"])
    df = df.reset_index(drop=True)
    print(df)
    df['Label'] = df['Label'].astype(int)

    columns_to_convert = ['Open', 'High', "Low", "Close", "Volume"]
    for col in columns_to_convert:
        df[col] = df[col].astype(float)
    df["Highest"] = np.where(df['Open'] > df['Close'], df['Open'], df['Close'])
    df["Swing_High"] = df["High"] - df['Highest']

    df["Lowest"] = np.where(df['Open'] > df['Close'], df['Close'], df['Open'])
    df["Swing_Low"] = df["Lowest"] - df["Low"]

    df = df.drop(columns=["Highest", "Lowest"])
    price = df['Close']

    df['EMA_150'] = calculate_ema(price, 150)
    df['EMA_200'] = calculate_ema(price, 200) 
    
    return df

def create_sequences(data, target, seq_length=10, prediction_length=3):
    X, y = [], []

    for i in range(len(data) - seq_length - prediction_length + 1):
        # Extract the input sequence
        seq_x = data[i:i+seq_length]
        seq_y = target[i+seq_length:i+seq_length+prediction_length]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def prepare_data(df,device, seq_length, prediction_length, label_column='Label', test_size=0.2, val_size=0.1):
    # Separate features and labels
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)

    x_processed, y_processed = create_sequences(X_scaled,y,seq_length=seq_length,prediction_length=prediction_length)
    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(x_processed, y_processed, test_size=(test_size + val_size))
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size))

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor