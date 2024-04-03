from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import os
import time
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, InputLayer
# from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from ModelTrainer import ModelTrainer  # Ensure this can handle Keras models if needed
from params import MODELS_DIR
from utils import ensure_1d_array

# Define a basic neural network for regression
# def create_nn_model(input_shape):
#     model = Sequential([
#         InputLayer(input_shape=(input_shape,)),
#         Dense(64, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
#     return model


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
        "XGBoost": XGBRegressor
        # "BasicNN": lambda: KerasRegressor(build_fn=create_nn_model, input_shape=6, epochs=100, batch_size=32, verbose=0)
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
