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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


def train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, hyperparameters=None):
    print(f"Training and evaluating {model_name}")
    if hyperparameters:
        model = GridSearchCV(model_class(), hyperparameters, scoring='neg_mean_squared_error', cv=5)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.squeeze()
        model.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {model.best_params_}")
        y_pred = model.predict(X_test)
    else:
        trainer = ModelTrainer(model_class())
        trainer.fit(X_train, y_train)
        y_pred = trainer.predict(X_test)
    ModelTrainer.evaluate(y_test, y_pred)
    return y_pred

if __name__ == '__main__':
    MODELS_DIR = 'models'
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv('./logs/500run2.csv')
    train_df = df.iloc[:1200]
    test_df = df.iloc[1200:]

    features = ['Function Name', 'Input Feature']
    targets = ['Duration', 'Max CPU Usage', 'Max Memory Usage']

    models = {
        "RandomForest": RandomForestRegressor,
        "LinearRegression": LinearRegression,
        "DecisionTree": DecisionTreeRegressor,
        "SVR": SVR,
        "XGBoost": XGBRegressor
    }

    hyperparameter_grids = {
        "RandomForest": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        "DecisionTree": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        "SVR": {"C": [0.1, 1, 10], "kernel": ["rbf"]},
        "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.01, 0.1]}
    }

    for target in targets:
        print(f"\n----- Evaluating for target: {target} -----")
        X_train, y_train = ModelTrainer(None).prepare_data(train_df, features, [target])
        X_test, y_test = ModelTrainer(None).prepare_data(test_df, features, [target])
        
        for model_name, model_class in models.items():
            hyperparameters = hyperparameter_grids.get(model_name, None)
            y_pred = train_and_evaluate_model(model_class, model_name, X_train, y_train, X_test, y_test, hyperparameters)