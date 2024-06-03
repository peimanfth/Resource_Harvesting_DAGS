class FunctionEntity:

    def __init__(self, dag_type, input_content=None, input_size=None, function_name=None,
                 function_input=None, duration=None, parallel_duration=None, 
                 memory_allocated=None, cpu_allocated=None, max_memory_usage=None,
                 max_cpu_usage=None, avg_cpu_usage=None, start_time=None, 
                 end_time=None, timeout_status=None, input_file=None, 
                 predicted_input_size=None, predicted_max_cpu_usage=None, 
                 predicted_max_memory_usage=None, stage=None):
        self.dag_type = dag_type
        self.input_content = input_content
        self.input_size = input_size
        self.function_name = function_name
        self.function_input = function_input
        self.duration = duration
        self.parallel_duration = parallel_duration
        self.predicted_max_cpu_usage = predicted_max_cpu_usage
        self.predicted_max_memory = predicted_max_memory_usage
        self.cpu_allocated = cpu_allocated
        self.memory_allocated = memory_allocated
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
        self.stage = stage
    
    #setters
    def set_input_size(self, input_size):
        self.input_size = input_size
    
    def set_predicted_input_size(self, predicted_input_size):
        self.predicted_input_size = predicted_input_size

    def set_predicted_max_cpu_usage(self, predicted_max_cpu_usage):
        self.predicted_max_cpu_usage = predicted_max_cpu_usage
    
    def set_predicted_max_memory_usage(self, predicted_max_memory_usage):
        self.predicted_max_memory_usage = predicted_max_memory_usage


    #getters
    def get_predicted_input_size(self):
        return self.predicted_input_size

    def get_predicted_max_cpu_usage(self):
        return self.predicted_max_cpu_usage

    def get_predicted_max_memory_usage(self):
        return self.predicted_max_memory_usage 


    def __repr__(self):
        return f"<FunctionEntity {self.function_name} CPU: {self.cpu_allocated} Mem: {self.memory_allocated}>"

    
