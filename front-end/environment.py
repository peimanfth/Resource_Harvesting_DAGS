from logger import Logger
import json
import time
from DAG import DAG
from redisClient import RedisClient
class Environment:

    def __init__(self, logger=None, redis_client=RedisClient()):
        self.dags = {}
        self.trace_settings = {
            'low': 0.1,
            'normal': 1.0,
            'high': 10.0
        }
        # self.logger = Logger().get_logger('Environment')
        self.logger = logger or Logger().get_logger('Environment')
        self.redis_client = redis_client

    def add_dag(self, dag_type, functions_data=None):
        dag = DAG(dag_type)
        self.dags[dag_type] = dag

    def generate_traces(self):
        traces = {'low': {}, 'normal': {}, 'high': {}}

        for trace_type, multiplier in self.trace_settings.items():
            for dag_type, dag in self.dags.items():
                self.logger.debug(f"Generating trace for {dag_type} with multiplier {multiplier}")
                traces[trace_type][dag_type] = self.generate_trace_for_dag(dag, multiplier)
        return traces

    def generate_trace_for_dag(self, dag, multiplier):
        trace_data = []
        num_functions = len(dag.functions)
        start_time = time.time()
        
        for i in range(0, len(dag.functions), num_functions):
            end_time = start_time + (num_functions * multiplier)
            trace_data.append({
                'dag_type': dag.dag_type,
                'start_time': start_time,
                'end_time': end_time,
                'invocation_rate': 1 / multiplier,
            })
            start_time = end_time
        return trace_data

if __name__ == "__main__":
    logger = Logger().get_logger('testing')
    
    env = Environment(logger=logger)
    env.add_dag('AS')
    env.add_dag('vid')
    # env.add_dag('CS')
    # env.add_dag('DS')

    traces = env.generate_traces()
    logger.info(json.dumps(traces, indent=4))
    
    dag = env.dags['AS']
    dag.set_input_file()
    logger.debug(dag.dag_type)
    logger.debug(dag.input_size)
    dag.PredictPipeline()
    for function in dag.functions:
        logger.info(function)
        logger.info(function.get_predicted_input_size())
        logger.info(function.get_predicted_max_cpu_usage())
        logger.info(function.get_predicted_max_memory_usage())
    # dag.invoke_DAG()