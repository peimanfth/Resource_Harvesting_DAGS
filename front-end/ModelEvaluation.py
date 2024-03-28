from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import os
from ModelTrainer import ModelTrainer
from params import MODELS_DIR
from utils import ensure_1d_array

# Define a function to train and evaluate a model
def train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test):
    print(f"Training and evaluating {model_name}")
    if model_name == "XGBoost":
        trainer = ModelTrainer(model_class())
    else:
        trainer = ModelTrainer(model_class())
    trainer.fit(X_train, y_train)
    y_pred = trainer.predict(X_test)
    trainer.evaluate(y_test, y_pred)
    # cpu_error_rate = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
    # mem_error_rate = ModelTrainer.calculate_memory_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
    # print(f"{model_name} CPU Usage Error Rate: {cpu_error_rate:.2%}")
    # print(f"{model_name} Memory Usage Error Rate: {mem_error_rate:.2%}")
    # Save the model and encoder
    trainer.save_model(os.path.join(MODELS_DIR, f'model_{model_name}.pkl'))
    trainer.save_encoder(os.path.join(MODELS_DIR, f'encoder_{model_name}.pkl'))
    return y_pred

if __name__ == '__main__':
    # Load your dataset
    df = pd.read_csv('./logs/Nimbus500_v1.csv')

    # Split your dataset
    train_df = df.iloc[:1200]
    test_df = df.iloc[1200:]

    # Define features and targets
    features = ['Function Name', 'Input Feature']
    targets = ['Duration', 'Max CPU Usage', 'Max Memory Usage']

    # Models to compare
    models = {
        "RandomForest": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "DecisionTree": DecisionTreeRegressor,
        "SVR": SVR,
        "XGBoost": XGBRegressor
    }

    results = {}

    for target in targets:
        print(f"\n----- Evaluating for target: {target} -----")
        X_train, y_train = ModelTrainer(None).prepare_data(train_df, features, [target])
        X_test, y_test = ModelTrainer(None).prepare_data(test_df, features, [target])
        
        for model_name, model_class in models.items():
            err = None
            y_pred = train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test)
            # Store results for further analysis if needed
            if target == "Max CPU Usage":
                cpu_err = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
                err = cpu_err
                print(f"{model_name} {target} CPU Usage Error Rate: {cpu_err:.2%}")
            elif target == "Max Memory Usage":
                mem_err = ModelTrainer.calculate_memory_usage_error_rate(ensure_1d_array(y_test), ensure_1d_array(y_pred))
                err = mem_err
                print(f"{model_name} {target} Memory Usage Error Rate: {mem_err:.2%}")
            
            results[f"{model_name}_{target}"] = {"y_test": y_test, "y_pred": y_pred, "error_rate": err}

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
