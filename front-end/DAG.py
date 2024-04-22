
from params import USER_CONFIG

class DAG:
    def __init__(self, dag_id):
        config = USER_CONFIG.get(dag_id, {})
        self.dag_id = dag_id
        self.memory = config.get('Memory', 0)
        self.cpu = config.get('CPU', 0)
        self.input_file = config.get('Params', '')
        self.functions = []

        for func_name, details in config.get('Functions', {}).items():
            self.add_function(
                function_name=func_name,
                user_memory=details.get('Memory', 0),
                user_cpu=details.get('CPU', 0),
                stage=details.get('stage', None)
            )

    def add_function(self, function_name, user_memory, user_cpu, stage):
        function_details = {
            'function_name': function_name,
            'user_memory': user_memory,
            'user_cpu': user_cpu,
            'stage': stage
        }
        self.functions.append(function_details)

    def set_input_file(self, input_file):
        self.input_file = input_file
    

# Example usage of the DAG class
dag = DAG('AS')

# Displaying DAG details for demonstration
print(f"DAG ID: {dag.dag_id}")
print(f"Overall Memory: {dag.memory}")
print(f"Overall CPU: {dag.cpu}")
print(f"Configuration File: {dag.input_file}")
for function in dag.functions:
    print(f"Function Name: {function['function_name']}")
    print(f"  Memory Allocated: {function['user_memory']}")
    print(f"  CPU Allocated: {function['user_cpu']}")
    print(f"  Stage: {function['stage']}")
