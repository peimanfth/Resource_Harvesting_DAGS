import statistics
import time
from queue import Queue
from threading import Thread, Event
from Monitor import monitor_peak

interval = 0.02

def handler(event, context=None):

    # monitor daemon part 1
    stop_signal = Event()
    q_cpu = Queue()
    q_mem = Queue()
    t = Thread(
        target=monitor_peak,
        args=(interval, q_cpu, q_mem, stop_signal),
        daemon=True
    )
    t.start()

    # Extract execution times and timestamps from the results
    results = event['value']
    execution_times = [result['latencies']['function_execution'] for result in results]
    start_times = [result['timestamps']['starting_time'] for result in results]
    finish_times = [result['timestamps']['finishing_time'] for result in results]

    # Calculate metrics
    avg_execution_time = statistics.mean(execution_times)
    median_execution_time = statistics.median(execution_times)
    min_execution_time = min(execution_times)
    max_execution_time = max(execution_times)
    percentile_95_execution_time = statistics.quantiles(execution_times, n=100)[94]  # 95th percentile
    
    # Total execution time (time from start of first function to end of last)
    total_execution_span = max(finish_times) - min(start_times)
    
    # Sum of all individual function execution times
    total_execution_time_sum = sum(execution_times)

    # Parallelism efficiency metrics
    # These metrics are useful to understand how well the parallel execution is performing.
    if total_execution_span > 0:
        efficiency = total_execution_time_sum / (len(execution_times) * total_execution_span)
    else:
        efficiency = 0  # Avoid division by zero
    

    # Create a summary of the metrics
    metrics_summary = {
        'average_execution_time': avg_execution_time,
        'median_execution_time': median_execution_time,
        'min_execution_time': min_execution_time,
        'max_execution_time': max_execution_time,
        '95th_percentile_execution_time': percentile_95_execution_time,
        'total_execution_span': total_execution_span,
        'total_execution_time_sum': total_execution_time_sum,
        'parallel_efficiency': efficiency
    }


    stop_signal.set()
    t.join()
    # monitor daemon part 2
    cpu_timestamp = []
    cpu_usage = []
    while q_cpu.empty() is False:
        (timestamp, cpu) = q_cpu.get()
        cpu_timestamp.append(timestamp)
        cpu_usage.append(cpu)

    mem_timestamp = []
    mem_usage = []
    while q_mem.empty() is False:
        (timestamp, mem) = q_mem.get()
        mem_timestamp.append(timestamp)
        mem_usage.append(mem)

    return {
        'metrics_summary' : metrics_summary,
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }

# Example usage:
# Assuming you have a list of results from your function instances
# results = [
#     {"latencies": {"function_execution": 0.5}, "timestamps": {"starting_time": 1609459200.0, "finishing_time": 1609459200.5}},
#     {"latencies": {"function_execution": 0.7}, "timestamps": {"starting_time": 1609459200.1, "finishing_time": 1609459200.8}},
#     # ... more results ...
# ]
