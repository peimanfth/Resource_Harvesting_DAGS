import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
import math
from utils import ensure_1d_array

class ModelTrainer:
    def __init__(self, model, encoder=None):
        self.model = model
        self.encoder = encoder if encoder else OneHotEncoder(sparse_output=False)

    def prepare_data(self, df, features, targets):
        X = df[features].fillna(0)  # Fill NaN values in features with 0
        y = df[targets].fillna(0)  # Fill NaN values in targets with 0

        # One-hot encode categorical features if necessary
        if 'Function Name'  in features and 'Function Input'in features:
            X_encoded = self.encoder.fit_transform(X[['Function Name']])
            X_encoded = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names_out(['Function Name']))
            X_encoded['Function Input'] = X['Function Input'].values
            X = X_encoded
        elif 'DAG Input Size' in features and 'Function Name' in features:
            X_encoded = self.encoder.fit_transform(X[['Function Name']])
            X_encoded = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names_out(['Function Name']))
            X_encoded['DAG Input Size'] = X['DAG Input Size'].values
            X = X_encoded

        return X, y
    
    def prepare_inference_data(self, df, features, x=None):
        X = df[features].fillna(0)

        # One-hot encode categorical features if necessary
        if 'Function Name' in features and 'DAG Input Size' in features:
            X_encoded = self.encoder.fit_transform(X[['Function Name']])
            X_encoded = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names_out(['Function Name']))
            X_encoded['DAG Input Size'] = X['DAG Input Size'].values
            X = X_encoded
        elif 'Predicted Input Size' in features and 'Function Name' in features:
            X_encoded = self.encoder.fit_transform(X[['Function Name']])
            X_encoded = pd.DataFrame(X_encoded, columns=self.encoder.get_feature_names_out(['Function Name']))
            X_encoded['Predicted Input Size'] = X['Predicted Input Size'].values
            X = X_encoded
        
        return X

    def fit(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def save_encoder(self, encoder_path):
        joblib.dump(self.encoder, encoder_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def load_encoder(self, encoder_path):
        self.encoder = joblib.load(encoder_path)

    @staticmethod
    def evaluate(y_true, y_pred):
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        print(f"MSE: {mse}")

    @staticmethod
    def calculate_cpu_usage_error_rate(y_actual, y_predicted):
        """
        Calculate the error rate for CPU usage predictions based on the criteria:
        If the predicted value falls in the integer range of the actual value, then it is not an error.
        
        Args:
            y_actual (array-like): The actual CPU usage values.
            y_predicted (array-like): The predicted CPU usage values.
            
        Returns:
            float: The error rate.
            list: The list of errors.
        """
        errors = 0
        error_list = []
        for actual, predicted in zip(y_actual, y_predicted):
            actualError = abs(actual - predicted)/actual
            error_list.append(actualError)
            # Check if predicted falls outside the integer range of actual
            if not (int(actual) <= predicted < int(actual) + 1):
                errors += 1
        
        error_rate = errors / len(y_actual)
        return error_rate,error_list
    


    @staticmethod
    def calculate_memory_usage_error_rate(y_actual, y_predicted):
        """
        Calculate the error rate for memory usage predictions based on the criteria:
        A prediction is considered an error if it falls between 2^n and 2^(n+1), where 2^n is the largest power of 2 that is less than or equal to the predicted value.
        
        Args:
            y_actual (array-like): The actual memory usage values.
            y_predicted (array-like): The predicted memory usage values.
            
        Returns:
            float: The error rate.
            list: The list of errors.
        """
        errors = 0
        error_list = []
        for actual, predicted in zip(y_actual, y_predicted):
            actualError = abs(actual - predicted)/actual
            error_list.append(actualError)
            n = math.floor(math.log2(predicted))
            if not (2**n <= actual <= 2**(n+1)):
                errors += 1
        
        error_rate = errors / len(y_actual)
        return error_rate, error_list
    
    @staticmethod
    def calculate_input_size_error_rate(y_actual, y_predicted):
        """
        Calculate the error rate for input size predictions based on the criteria:
        If the predicted value falls in the integer range of the actual value, then it is not an error.
        
        Args:
            y_actual (array-like): The actual input size values.
            y_predicted (array-like): The predicted input size values.
            
        Returns:
            float: The error rate.
            list: The list of errors.
        """
        errors = 0
        error_list = []
        for actual, predicted in zip(y_actual, y_predicted):
            actualError = abs(actual - predicted)/actual
            error_list.append(actualError)
            # Check if predicted falls outside the 200 integer range of actual
            if not (int(actual) - 500 <= predicted < int(actual) + 500):
                errors += 1
        
        error_rate = errors / len(y_actual)
        return error_rate,error_list



    
if __name__ == '__main__':
    from sklearn.ensemble import RandomForestRegressor
    import os

    # Create 'models' directory if it does not exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)


    # Assuming ModelTrainer class definition is already provided as above

    # Load your dataset
    df = pd.read_csv('./logs/Nimbus500_v1.csv')

    # Split your dataset
    train_df = df.iloc[:1200]
    test_df = df.iloc[1200:]

    # Define features and targets
    features = ['Function Name', 'Input Feature']
    target_duration = 'Duration'
    target_cpu_usage = 'Max CPU Usage'
    target_memory_usage = 'Max Memory Usage'

    # Initialize the model trainers for each prediction target
    trainer_duration = ModelTrainer(RandomForestRegressor(random_state=42))
    trainer_cpu_usage = ModelTrainer(RandomForestRegressor(random_state=42))
    trainer_memory_usage = ModelTrainer(RandomForestRegressor(random_state=42))

    # Prepare the data for each target
    X_train, y_train_duration = trainer_duration.prepare_data(train_df, features, [target_duration])
    X_test, y_test_duration = trainer_duration.prepare_data(test_df, features, [target_duration])

    _, y_train_cpu_usage = trainer_cpu_usage.prepare_data(train_df, features, [target_cpu_usage])
    _, y_test_cpu_usage = trainer_cpu_usage.prepare_data(test_df, features, [target_cpu_usage])

    _, y_train_memory_usage = trainer_memory_usage.prepare_data(train_df, features, [target_memory_usage])
    _, y_test_memory_usage = trainer_memory_usage.prepare_data(test_df, features, [target_memory_usage])

    # Fit the models
    trainer_duration.fit(X_train, y_train_duration)
    trainer_cpu_usage.fit(X_train, y_train_cpu_usage)
    trainer_memory_usage.fit(X_train, y_train_memory_usage)

    # Predict and evaluate
    y_pred_duration = trainer_duration.predict(X_test)
    trainer_duration.evaluate(y_test_duration, y_pred_duration)

    y_pred_cpu = trainer_cpu_usage.predict(X_test)
    trainer_cpu_usage.evaluate(y_test_cpu_usage, y_pred_cpu)

    y_pred_memory = trainer_memory_usage.predict(X_test)
    trainer_memory_usage.evaluate(y_test_memory_usage, y_pred_memory)
    error_rate, error_list = ModelTrainer.calculate_memory_usage_error_rate(ensure_1d_array(y_test_memory_usage),ensure_1d_array(y_pred_memory))
    print(f"Memory Usage Error Rate: {error_rate}")
    #write the error list to a csv file
    error_df = pd.DataFrame(error_list)
    error_df.to_csv('memory_error_list.csv', index=False)


    trainer_duration.save_model(os.path.join(models_dir, 'model_duration.pkl'))
    trainer_duration.save_encoder(os.path.join(models_dir, 'encoder_duration.pkl'))

    trainer_cpu_usage.save_model(os.path.join(models_dir, 'model_cpu_usage.pkl'))
    trainer_cpu_usage.save_encoder(os.path.join(models_dir, 'encoder_cpu_usage.pkl'))
    # print(f"CPU Usage Error Rate: {ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test_cpu_usage),ensure_1d_array(y_pred_cpu))}")
    error_rate, error_list = ModelTrainer.calculate_cpu_usage_error_rate(ensure_1d_array(y_test_cpu_usage),ensure_1d_array(y_pred_cpu))
    print(f"CPU Usage Error Rate: {error_rate}")
    #write the error list to a csv file
    error_df = pd.DataFrame(error_list)
    error_df.to_csv('cpu_error_list.csv', index=False)


    trainer_memory_usage.save_model(os.path.join(models_dir, 'model_memory_usage.pkl'))
    trainer_memory_usage.save_encoder(os.path.join(models_dir, 'encoder_memory_usage.pkl'))

