
# import time
from params import USER_CONFIG, size_model, mem_model, cpu_model
import pandas as pd
from ModelTrainer import ModelTrainer
from FunctionEntity import FunctionEntity

class DAG:
    
    def __init__(self, dag_id):
        config = USER_CONFIG.get(dag_id, {})
        self.dag_id = dag_id
        self.memory = config.get('Memory', 0)
        self.cpu = config.get('CPU', 0)
        self.input_file = config.get('Params', '')
        self.input_size = None
        self.functions = []

        for func_name, details in config.get('Functions', {}).items():
            self.add_function(
                function_name=func_name,
                user_memory=details.get('Memory', 0),
                user_cpu=details.get('CPU', 0),
                stage=details.get('stage', None)
            )

    def add_function(self, function_name, user_memory, user_cpu, stage):
        function_entity = FunctionEntity(
            dag_id=self.dag_id,
            function_name=function_name,
            memory_allocated=user_memory,
            cpu_allocated=user_cpu,
            # Add other parameters as needed
        )
        self.functions.append(function_entity)



    def PredictPipeline(self):

        FunctionsList = self.get_function_names()
        DAGinputSizeList = [self.get_input_size()] * len(FunctionsList)
        dict = {'Function Name': FunctionsList, 'DAG Input Size': DAGinputSizeList}
        print(dict)
        request = pd.DataFrame(dict)

        features = ['DAG Input Size', 'Function Name']
        # target = 'Function Input'


        X_new = ModelTrainer(None).prepare_inference_data(request, features)

        request['Predicted Input Size'] = size_model.predict(X_new)
        X_new['Predicted Input Size'] = request['Predicted Input Size'].values
            
        # Assuming 'ModelTrainer' or similar utility for data preparation is available
        # Prepare data (ensure the feature names and model input match training setup)
        features = ['Function Name', 'Predicted Input Size']  
        # target = 'Max CPU Usage'
        X_new = ModelTrainer(None).prepare_inference_data(request, features)

        #change column name from  Predicted Input Size to Function Input
        X_new = X_new.rename(columns = {'Predicted Input Size':'Function Input'})
        request['Predicted Max CPU Usage'] = cpu_model.predict(X_new)
        request['Predicted Max Memory Usage'] = mem_model.predict(X_new)
        for function in self.functions:
            
            function.set_predicted_input_size(request['Predicted Input Size'][request['Function Name'] == function.function_name].values[0])
            function.set_predicted_max_cpu_usage(request['Predicted Max CPU Usage'][request['Function Name'] == function.function_name].values[0])
            function.set_predicted_max_memory_usage(request['Predicted Max Memory Usage'][request['Function Name'] == function.function_name].values[0])
    
    #setters

    def set_input_file(self, input_file):
        self.input_file = input_file

    def set_input_size(self, input_size):
        self.input_size = input_size

    #getters

    def get_function_names(self):
        return [function.function_name for function in self.functions]
    

    def get_input_size(self):
        return self.input_size

if __name__ == "__main__":
    dag = DAG('AS')
    dag.set_input_size(5000)
    dag.PredictPipeline()
    for function in dag.functions:
        print(function)
        print(function.get_predicted_input_size())
        print(function.get_predicted_max_cpu_usage())
        print(function.get_predicted_max_memory_usage())


# dag = DAG('AS')



# print(f"DAG ID: {dag.dag_id}")
# print(f"Overall Memory: {dag.memory}")
# print(f"Overall CPU: {dag.cpu}")
# print(f"Configuration File: {dag.input_file}")
# for function in dag.functions:
#     print(f"Function Name: {function.function_name}")
#     print(f"  Memory Allocated: {function.memory_allocated}")
#     print(f"  CPU Allocated: {function.cpu_allocated}")
#     print(f"  Stage: {function.stage}")
# # list of all function names
# print(dag.get_function_names())
# print(dag.functions)
