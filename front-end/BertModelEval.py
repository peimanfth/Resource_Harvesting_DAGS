from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import os
import time
from sklearn.preprocessing import StandardScaler
from ModelTrainer import ModelTrainer  # Ensure this can handle Keras models if needed
from params import MODELS_DIR
from utils import ensure_1d_array

# Define a basic neural network for regression
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Define a function to train and evaluate a model
# def train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test):
#     print(f"Training and evaluating {model_name}")
#     if model_name == "XGBoost":
#         trainer = ModelTrainer(model_class())
#     else:
#         trainer = ModelTrainer(model_class())
#     trainer.fit(X_train, y_train)
#     y_pred = trainer.predict(X_test)
#     trainer.evaluate(y_test, y_pred)
#     # cpu_error_rate = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
#     # mem_error_rate = ModelTrainer.calculate_memory_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
#     # print(f"{model_name} CPU Usage Error Rate: {cpu_error_rate:.2%}")
#     # print(f"{model_name} Memory Usage Error Rate: {mem_error_rate:.2%}")
#     # Save the model and encoder
#     trainer.save_model(os.path.join(MODELS_DIR, f'model_{model_name}.pkl'))
#     trainer.save_encoder(os.path.join(MODELS_DIR, f'encoder_{model_name}.pkl'))
#     return y_pred
def train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test):
    print(f"Training and evaluating {model_name}")
    start_time = time.time()
    trainer = ModelTrainer(model_class())
    trainer.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = trainer.predict(X_test)
    inference_time = time.time() - start_time
    
    trainer.evaluate(y_test, y_pred)
    trainer.save_model(os.path.join(MODELS_DIR, f'model_{model_name}.pkl'))
    trainer.save_encoder(os.path.join(MODELS_DIR, f'encoder_{model_name}.pkl'))
    
    return y_pred, training_time, inference_time

def train_and_evaluate_model_pytorch(model, model_name, X_train, y_train, X_test, y_test):
    print(f"Training and evaluating {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    start_time = time.time()
    for epoch in range(100):  # Assuming 100 epochs
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.view(-1, 1))
        loss.backward()
        optimizer.step()
    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
    inference_time = time.time() - start_time - training_time
    
    # Assuming your ModelTrainer or another utility function handles the evaluation metrics
    # For saving model, you might want to adjust for PyTorch specifics
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'model_{model_name}.pth'))
    
    return y_pred.cpu().numpy(), training_time, inference_time

if __name__ == '__main__':
    

    # Define the path to the new dataset
    dataset_path = './logs/parallelFunctionsBERT.csv'

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Assuming a 768-dimensional embedding plus encoded function names
    input_size = df.shape[1] - 1  # Excluding the target column

    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Define the models to compare
    models = {
        "RandomForest": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "DecisionTree": DecisionTreeRegressor,
        "SVR": SVR,
        "XGBoost": XGBRegressor,
        # Adjust the input size for the neural network based on the actual feature size
        "BasicNN": SimpleNN(input_size=input_size)
    }

    # Preparing the feature matrix (X) and the target (y)
    X_train = train_df.drop('Max CPU Usage', axis=1)
    y_train = train_df['Max CPU Usage']
    X_test = test_df.drop('Max CPU Usage', axis=1)
    y_test = test_df['Max CPU Usage']

    results = {}
    error_directory = "./logsBert/errors"
    os.makedirs(error_directory, exist_ok=True)

    time_results = []

    for model_name, model in models.items():
        print(f"\n----- Training and Evaluating {model_name} -----")
        if model_name == "BasicNN":
            # Adjust the training function for PyTorch model as per your implementation
            y_pred, training_time, inference_time = train_and_evaluate_model_pytorch(model, model_name, X_train, y_train, X_test, y_test)
            cpu_err, cpu_err_list = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
            error_df = pd.DataFrame(cpu_err_list)
            # write the error list to a csv file in the error directory
            error_df.to_csv(f'{error_directory}/{model_name}_cpu_error_list.csv', index=False)
            err = cpu_err
            print(f"{model_name} MAX CPU Usage Error Rate: {cpu_err:.2%}")
        else:
            # Your existing training and evaluation function for scikit-learn models
            y_pred, training_time, inference_time = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
        
        mse = mean_squared_error(y_test, y_pred)
        print(f"{model_name} Mean Squared Error: {mse}")
        cpu_err, cpu_err_list = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
        error_df = pd.DataFrame(cpu_err_list)
        # write the error list to a csv file in the error directory
        error_df.to_csv(f'{error_directory}/{model_name}_cpu_error_list.csv', index=False)
        err = cpu_err
        print(f"{model_name} MAX CPU Usage Error Rate: {cpu_err:.2%}")
        
        # Record the training and inference times
        time_results.append({
            "Model": model_name,
            "MSE": mse,
            "Training Time": training_time,
            "Inference Time": inference_time,
        })

    # Save the time and performance results
    time_df = pd.DataFrame(time_results)
    os.makedirs('./logsBert/times', exist_ok=True)
    time_df.to_csv('./logsBert/times/model_performance_statistics.csv', index=False)

