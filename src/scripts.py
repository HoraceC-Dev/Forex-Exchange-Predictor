import torch
import torch.nn as nn

from data_processing import get_data,prepare_data
from model import CustomModel
from train import train_model
from evaluate import evaluate_model

# Load and preprocess data
df = get_data("dataset/GBPUSD_H4_data.csv")

# Prepare data for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, device)

input_size = X_train.shape[2]
hidden_size = 64
output_size = 3
num_layers = 1
prediction_length = 3

learning_rate = 0.001

# Define model, criterion, and optimizer
model = CustomModel(input_size, hidden_size, output_size, num_layers,prediction_length)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 20

# Train the model
best_model_state = train_model(model, X_train, y_train, X_val, y_val, criterion, optimizer, num_epochs, device)

# Evaluate the model
model.load_state_dict(best_model_state)
evaluate_model(model, X_test, y_test, criterion, [-1,0,1], device)
model_save_path = 'model/model.pth'
torch.save(model.state_dict(), model_save_path)