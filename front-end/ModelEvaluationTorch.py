from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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
    df = pd.read_csv('./logs/500run2.csv')

    train_df = df.iloc[:1200]
    test_df = df.iloc[1200:]

    features = ['Function Name', 'Input Feature']
    targets = ['Duration', 'Max CPU Usage', 'Max Memory Usage']
    

    # Models to compare
    models = {
        "RandomForest": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "DecisionTree": DecisionTreeRegressor,
        "SVR": SVR,
        "XGBoost": XGBRegressor,
        "BasicNN": SimpleNN(input_size=6)
    }

    results = {}
    error_directory = "./logs/errors"
    os.makedirs(error_directory, exist_ok=True)

    time_results = []

    for target in targets:
        print(f"\n----- Evaluating for target: {target} -----")
        X_train, y_train = ModelTrainer(None).prepare_data(train_df, features, [target])
        X_test, y_test = ModelTrainer(None).prepare_data(test_df, features, [target])
        
        for model_name, model_class in models.items():
            err = None
            start_time = time.time()
            if model_name == "BasicNN":
                y_pred, training_time, inference_time = train_and_evaluate_model_pytorch(model_class, model_name, X_train, y_train, X_test, y_test)
            else:
                y_pred, training_time, inference_time = train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test)
            total_time = time.time() - start_time
            if target == "Max CPU Usage":
                cpu_err, cpu_err_list = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
                error_df = pd.DataFrame(cpu_err_list)
                # write the error list to a csv file in the error directory
                error_df.to_csv(f'{error_directory}/{model_name}_cpu_error_list.csv', index=False)
                err = cpu_err
                print(f"{model_name} {target} CPU Usage Error Rate: {cpu_err:.2%}")
            elif target == "Max Memory Usage":
                mem_err, mem_err_list = ModelTrainer.calculate_memory_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
                error_df = pd.DataFrame(mem_err_list)
                # write the error list to a csv file in the error directory
                error_df.to_csv(f'{error_directory}/{model_name}_memory_error_list.csv', index=False)
                err = mem_err
                print(f"{model_name} {target} Memory Usage Error Rate: {mem_err:.2%}")

            time_results.append({
                "Model": model_name,
                "Target": target,
                "Training Time": training_time,
                "Inference Time": inference_time,
                "Total Time": total_time
            })
            results[f"{model_name}_{target}"] = {"y_test": y_test, "y_pred": y_pred, "error_rate": err}
    time_df = pd.DataFrame(time_results)
    os.makedirs('./logs/times', exist_ok=True)
    time_df.to_csv('./logs/times/model_time_statistics.csv', index=False)
    # for target in targets:
    #     print(f"\n----- Evaluating for target: {target} -----")
    #     X_train, y_train = ModelTrainer(None).prepare_data(train_df, features, [target])
    #     X_test, y_test = ModelTrainer(None).prepare_data(test_df, features, [target])
        
    #     for model_name, model_class in models.items():
    #         print(f"\nTraining and evaluating {model_name} for {target}")
    #         trainer = ModelTrainer(model_class(random_state=42) if model_name != "SVR" else model_class())
    #         trainer.fit(X_train, y_train)
    #         y_pred = trainer.predict(X_test)
    #         trainer.evaluate(y_test, y_pred)
    #         error_rate = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
    #         print(f"{model_name} {target} CPU Usage Error Rate: {error_rate:.2%}")
    #         # Save the model and encoder
    #         trainer.save_model(os.path.join(models_dir, f'model_{model_name}_{target}.pkl'))
    #         trainer.save_encoder(os.path.join(models_dir, f'encoder_{model_name}_{target}.pkl'))
    #         # Store results for further analysis if needed
    #         results[f"{model_name}_{target}"] = {"y_test": y_test, "y_pred": y_pred, "error_rate": error_rate}
