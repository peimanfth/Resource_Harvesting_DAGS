import time
import logging
import csv
import psutil
from threading import Thread, Event
from DAG import DAG
from logger import Logger
from redisClient import RedisClient  # Assuming you have a RedisClient class
import os
import json
from Couch import Couch
from params import ACTIVATIONS_DB_NAME

class DAGEnvironment:
    def __init__(self, dag_type='AS', log_name='testing'):
        self.logger = Logger().get_logger(log_name)
        self.dag = DAG(dag_type, logger=self.logger)
        self.redis_client = RedisClient(logger=self.logger)  # Initialize your Redis client
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.stop_event = Event()
        self.activations = {}
        self.start_time = None
        self.couch = Couch()

    def set_input_file(self, index):
        input_file = f'/home/peiman/openwhisk/front-end/inputs/experiment/run{index+1}.json'
        # input_file = f'/home/peiman/openwhisk/front-end/inputs/experiment/run{1}.json'
        self.dag.set_input_file(input_file)
        return input_file

    def process_request(self, index):
        input_file = self.set_input_file(index)
        start_time = time.time()
        self.dag.PredictPipeline()
        predict_pipeline_time = time.time() - start_time
        self.logger.info(f"PredictPipeline time: {predict_pipeline_time} seconds")

        for function in self.dag.functions:
            self.logger.info(function)
            self.logger.info(function.get_predicted_input_size())
            self.logger.info(function.get_predicted_max_cpu_usage())
            self.logger.info(function.get_predicted_max_memory_usage())
        self.dag.set_dag_utilization_to_redis()
        self.dag.log_dag_utilization()
        activation = self.dag.invoke_DAG()
        return activation, input_file

    def check_undone_requests(self):
        value = self.redis_client.get_value("n_undone_request")
        return int(value) == 0 if value is not None else False

    def set_undone_requests(self, n=1):
        self.redis_client.set_value("n_undone_request", n)

    def monitor_resources(self):
        while not self.stop_event.is_set():
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            self.memory_usage.append(psutil.virtual_memory().percent)
            self.timestamps.append(time.time() - self.start_time)  # Relative timestamp
            time.sleep(0.1)  # Monitor every 0.1 second


    def save_resource_usage_to_csv(self, algorithm_name, num_requests, requests_per_minute):
        directory = '/home/peiman/Desktop/explogs1'
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(directory, f'resource_usage_{algorithm_name}_{num_requests}req_{requests_per_minute}rpm.csv')

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Relative Timestamp (s)', 'CPU Usage (%)', 'Memory Usage (%)'])
            for timestamp, cpu, memory in zip(self.timestamps, self.cpu_usage, self.memory_usage):
                writer.writerow([timestamp, cpu, memory])

    def get_activation_duration(self, activation_id):
        doc = self.couch.get_doc_with_id(ACTIVATIONS_DB_NAME, activation_id)
        return doc.get('duration', None)
    def get_parallel_stage_duration(self, activation_id):
        """
        Retrieves the parallel stage duration for the given activation ID.
        
        Parameters:
        - activation_id: The activation ID for which the parallel stage duration is to be fetched.
        
        Returns:
        - The parallel stage duration in milliseconds, or None if not found.
        """
        return self.couch.get_parallel_stage_duration(ACTIVATIONS_DB_NAME, activation_id)

    def save_experiment_results_to_json(self, algorithm_name, num_requests, requests_per_minute, total_time):
        directory = '/home/peiman/Desktop/explogs1'
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(directory, f'experiment_results_{algorithm_name}_{num_requests}req_{requests_per_minute}rpm.json')
        
        results = {
            'algorithm_name': algorithm_name,
            'num_requests': num_requests,
            'requests_per_minute': requests_per_minute,
            'total_time': total_time,
            'activations': {}
        }

        for request_id, activation_info in self.activations.items():
            activation_id = activation_info['activation']
            input_file = activation_info['input_file']
            duration = self.get_activation_duration(activation_id)
            parallel_stage_duration = self.get_parallel_stage_duration(activation_id)  # Fetch the parallel stage duration
            
            results['activations'][request_id] = {
                'activationId': activation_id,
                'duration': duration,
                'parallel_stage_duration': parallel_stage_duration,  # Add parallel stage duration here
                'input_file': input_file
            }

        with open(filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)


    def run_experiment(self, num_requests=100, requests_per_minute=10, algorithm_name='JSQ'):
        self.set_undone_requests(num_requests)
        interval = 60 / requests_per_minute  # Interval between requests in seconds
        # Start the resource monitoring thread
        monitor_thread = Thread(target=self.monitor_resources)
        monitor_thread.start()
        self.start_time = time.time()


        for i in range(num_requests):
            activation, input_file = self.process_request(i)
            self.activations[f'request_{i}'] = {'activation': activation, 'input_file': input_file}
            if i < num_requests - 1:  # Sleep between requests, but not after the last one
                time.sleep(interval)
        
        # Wait until all requests are processed
        while not self.check_undone_requests():
            time.sleep(0.1)  # Check every second
        end_time = time.time()

        # Stop the resource monitoring thread
        self.stop_event.set()
        monitor_thread.join()

        # Log the end-to-end time
        total_time = end_time - self.start_time
        self.logger.info(f"End-to-end time for {num_requests} requests: {total_time} seconds")

        # Save resource usage to CSV
        self.save_resource_usage_to_csv(algorithm_name, num_requests, requests_per_minute)
        time.sleep(5)
        # Save experiment results to JSON
        self.save_experiment_results_to_json(algorithm_name, num_requests, requests_per_minute, total_time)
        # Log activations
        self.logger.info(f"Activations: {self.activations}")

if __name__ == "__main__":
    # Ensure the inputs directory exists
    os.makedirs('/home/peiman/openwhisk/front-end/inputs/experiment', exist_ok=True)

    env = DAGEnvironment()
    env.run_experiment(num_requests=180, requests_per_minute=15, algorithm_name='GreedyJSQ')
    time.sleep(5)
    env = DAGEnvironment()
    env.run_experiment(num_requests=180, requests_per_minute=30, algorithm_name='GreedyJSQ')
    time.sleep(5)
    env = DAGEnvironment()
    env.run_experiment(num_requests=180, requests_per_minute=60, algorithm_name='GreedyJSQ')
  