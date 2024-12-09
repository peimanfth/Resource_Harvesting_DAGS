
# import time
from params import USER_CONFIG, size_model, mem_model, cpu_model, WSK_CLI, WSK_ACTION, MEMORY_CAP_PER_FUNCTION, CPU_CAP_PER_FUNCTION, MEM_UNIT, CPU_UNIT
from utils import run_shell_command
import pandas as pd
from ModelTrainer import ModelTrainer
from FunctionEntity import FunctionEntity
from logger import Logger
from redisClient import RedisClient
import json
import math

class DAG:
    
    def __init__(self, dag_type, logger=None):
        config = USER_CONFIG.get(dag_type, {})
        self.dag_type = dag_type
        self.memory = config.get('Memory', 0)
        self.cpu = config.get('CPU', 0)
        self.totalMemory =  None
        self.totalCPU = None
        self.input_file = config.get('Params', '')
        self.input_size = None
        self.functions = []
        self.logger = logger
        for func_name, details in config.get('Functions', {}).items():
            self.add_function(
                function_name=func_name,
                user_memory=details.get('Memory', 0),
                user_cpu=details.get('CPU', 0),
                stage=details.get('stage', None)
            )

    def add_function(self, function_name, user_memory, user_cpu, stage):
        function_entity = FunctionEntity(
            dag_type=self.dag_type,
            function_name=function_name,
            memory_allocated=user_memory,
            cpu_allocated=user_cpu,
            # Add other parameters as needed
        )
        self.functions.append(function_entity)

    def invoke_DAG(self):
        container_ID, error = run_shell_command(f"{WSK_CLI} {WSK_ACTION} invoke {self.dag_type} -P {self.input_file} | awk '{{print $6}}'")
        # Omit tabs, newlines, and spaces
        container_ID = container_ID.replace('\n', '').replace('\t', '').replace(' ', '')
        print(container_ID)
        if error:
            print("Errors:", error)
        return container_ID


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
            function.set_predicted_max_cpu_usage(math.ceil(request['Predicted Max CPU Usage'][request['Function Name'] == function.function_name].values[0]))
            function.set_predicted_max_memory_usage((math.ceil(request['Predicted Max Memory Usage'][request['Function Name'] == function.function_name].values[0] / MEM_UNIT)) * MEM_UNIT)


        self.totalCPU = sum([function.cpu_allocated for function in self.functions])
        self.totalMemory = sum([function.memory_allocated for function in self.functions])

    
    #setters

    def set_input_file(self, input_file='inputs/AS1.json'):
        self.input_file = input_file
        with open(input_file, 'r') as f:
            data = json.load(f)
        if self.dag_type == 'AS':
            self.input_size = data.get('data', {}).get('num_of_iterations', None)

    def set_input_size(self, input_size):
        self.input_size = input_size


    def set_dag_utilization_to_redis(self):
        redis_client = RedisClient(logger=self.logger)
        redis_client.connect()
        for function in self.functions:
            redis_client.set_dag_utilization(
                function.function_name,
                function.get_predicted_max_cpu_usage(),
                function.get_predicted_max_memory_usage()
            )
        redis_client.close()

    def log_dag_utilization(self):
        for function in self.functions:
            self.logger.info(f"Function: {function.function_name}")
            self.logger.info(f"  Predicted Max CPU Usage: {function.get_predicted_max_cpu_usage()}")
            self.logger.info(f"  Predicted Max Memory Usage: {function.get_predicted_max_memory_usage()}")

    #getters

    def get_function_names(self):
        return [function.function_name for function in self.functions]
    

    def get_input_size(self):
        return self.input_size

if __name__ == "__main__":
    logger = Logger().get_logger('testing')
    dag = DAG('AS', logger=logger)
    dag.set_input_file()
    logger.debug(dag.dag_type)
    logger.debug(dag.input_size)
    dag.PredictPipeline()
    for function in dag.functions:
        logger.info(function)
        logger.info(function.get_predicted_input_size())
        logger.info(function.get_predicted_max_cpu_usage())
        logger.info(function.get_predicted_max_memory_usage())
    dag.set_dag_utilization_to_redis()
    dag.log_dag_utilization()
    # dag.invoke_DAG()
    


# dag = DAG('AS')



# print(f"DAG ID: {dag.dag_type}")
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
