
from params import USER_CONFIG

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
        function_details = {
            'function_name': function_name,
            'user_memory': user_memory,
            'user_cpu': user_cpu,
            'stage': stage
        }
        self.functions.append(function_details)

    def set_input_file(self, input_file):
        self.input_file = input_file

    def set_input_size(self, input_size):
        self.input_size = input_size
class FunctionEntity:
    def __init__(self, dag_id, input_content=None, input_size=None, function_name=None,
                 function_input=None, duration=None, parallel_duration=None, 
                 memory_allocated=None, cpu_allocated=None, max_memory_usage=None,
                 max_cpu_usage=None, avg_cpu_usage=None, start_time=None, 
                 end_time=None, timeout_status=None, input_file=None, 
                 predicted_input_size=None, predicted_max_cpu_usage=None, 
                 predicted_max_memory_usage=None):
        self.dag_id = dag_id
        self.input_content = input_content
        self.input_size = input_size
        self.function_name = function_name
        self.function_input = function_input
        self.duration = duration
        self.parallel_duration = parallel_duration
        self.memory_allocated = memory_allocated
        self.cpu_allocated = cpu_allocated
        self.max_memory_usage = max_memory_usage
        self.max_cpu_usage = max_cpu_usage
        self.avg_cpu_usage = avg_cpu_usage
        self.start_time = start_time
        self.end_time = end_time
        self.timeout_status = timeout_status
        self.input_file = input_file
        self.predicted_input_size = predicted_input_size
        self.predicted_max_cpu_usage = predicted_max_cpu_usage
        self.predicted_max_memory_usage = predicted_max_memory_usage

    def __repr__(self):
        return f"<FunctionEntity {self.function_name} CPU: {self.cpu_allocated} Mem: {self.memory_allocated}>"

    


dag = DAG('AS')


print(f"DAG ID: {dag.dag_id}")
print(f"Overall Memory: {dag.memory}")
print(f"Overall CPU: {dag.cpu}")
print(f"Configuration File: {dag.input_file}")
for function in dag.functions:
    print(f"Function Name: {function['function_name']}")
    print(f"  Memory Allocated: {function['user_memory']}")
    print(f"  CPU Allocated: {function['user_cpu']}")
    print(f"  Stage: {function['stage']}")
