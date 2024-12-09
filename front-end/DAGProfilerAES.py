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
from params import MODELS_DIR_AES as MODELS_DIR

def ensure_1d_array(data):
        """
        Ensure the data is a 1-dimensional numpy array.
        
        This function checks if the input data is a pandas DataFrame or Series, 
        or a numpy ndarray, and then converts it to a 1-dimensional numpy array.
        """
        if isinstance(data, pd.DataFrame):
            # If the DataFrame has more than one column, raise an error
            if data.shape[1] != 1:
                raise ValueError("DataFrame must have exactly one column.")
            data = data.iloc[:, 0].values  # Convert the single column to a numpy array
        elif isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1:
            data = data.flatten()
        
        # Check if the data is still not 1-dimensional
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("Data could not be converted to a 1-dimensional array.")
        
        return data

# Define a basic neural network for regression
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, target):
    print(f"Training and evaluating {model_name}")
    start_time = time.time()
    trainer = ModelTrainer(model_class())
    trainer.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = trainer.predict(X_test)
    inference_time = time.time() - start_time
    
    trainer.evaluate(y_test, y_pred)
    trainer.save_model(os.path.join(MODELS_DIR, f'model_{model_name}_{target}.pkl'))
    # trainer.save_encoder(os.path.join(MODELS_DIR, f'encoder_{model_name}.pkl'))
    
    return y_pred, training_time, inference_time

def train_and_evaluate_model_pytorch(model, model_name, X_train, y_train, X_test, y_test, target):
    print(f"Training and evaluating {model_name} for {target}")
    
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
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.view(-1, 1))
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    training_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
    inference_time = time.time() - start_time - training_time
    
    # Assuming your ModelTrainer or another utility function handles the evaluation metrics
    # For saving model, you might want to adjust for PyTorch specifics
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'model_{model_name}_{target}.pth'))
    
    return y_pred.cpu().numpy(), training_time, inference_time

def inputPrediction(df, models):
 
    train_df = df.sample(frac=0.8, random_state=42)
    print(train_df.shape)
    print(train_df.shape)
    test_df = df.drop(train_df.index)
    print(test_df.shape)

    features = ['DAG Input Size', 'Function Name']

    # targets = ['Duration', 'Max CPU Usage', 'Max Memory Usage']
    targets = ['Function Input']
    
    results = {}
    error_directory = "./logs/aes/errors"
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
                y_pred, training_time, inference_time = train_and_evaluate_model_pytorch(model_class, model_name, X_train, y_train, X_test, y_test,target)
            else:
                y_pred, training_time, inference_time = train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, target)
            total_time = time.time() - start_time

            input_size_err, input_size_err_lst = ModelTrainer.calculate_input_size_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
            error_df = pd.DataFrame(input_size_err_lst)
            # write the error list to a csv file in the error directory
            error_df.to_csv(f'{error_directory}/{model_name}_input_size_error_list.csv', index=False)
            err = input_size_err
            print(f"{model_name} {target} Input size Error Rate: {input_size_err:.2%}")
            time_results.append({
                "Model": model_name,
                "Target": target,
                "Training Time": training_time,
                "Inference Time": inference_time,
                "Total Time": total_time
            })
            results[f"{model_name}_{target}"] = {"y_test": y_test, "y_pred": y_pred, "error_rate": err}
    time_df = pd.DataFrame(time_results)
    os.makedirs('./logs/aes/times', exist_ok=True)
    time_df.to_csv('./logs/aes/times/model_time_statistics_InputSize.csv', index=False)

def utilPrediction(df, models):
  
    train_df = df.sample(frac=0.8, random_state=42)
    print(train_df.shape)
    test_df = df.drop(train_df.index)
    print(test_df.shape)

    features = ['Function Name', 'Function Input']

    # targets = ['Duration', 'Max CPU Usage', 'Max Memory Usage']
    targets = ['Max CPU Usage', 'Max Memory Usage']
    

    results = {}
    error_directory = "./logs/aes/errors"
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
                y_pred, training_time, inference_time = train_and_evaluate_model_pytorch(model_class, model_name, X_train, y_train, X_test, y_test,target)
            else:
                y_pred, training_time, inference_time = train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, target)
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
    os.makedirs('./logs/aes/times', exist_ok=True)
    time_df.to_csv('./logs/aes/times/model_time_statistics_Utilization.csv', index=False)
   

if __name__ == '__main__':
    # The profiling data. We should generate distinct profiling data for each DAG
    df = pd.read_csv('/home/peiman/openwhisk/front-end/logs/2024-10-17_12-17-55/test.csv')



    # Models to compare
    models = {
        "RandomForest": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "DecisionTree": DecisionTreeRegressor,
        "SVR": SVR,
        "XGBoost": XGBRegressor,
        #should be nodes+1
        "BasicNN": SimpleNN(input_size=6)
    }
    inputPrediction(df, models)

    utilPrediction(df, models)