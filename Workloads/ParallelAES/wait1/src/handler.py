import time
import random
import string
import pyaes
from multiprocessing import Process, Manager, cpu_count
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

    time.sleep(1)

    stop_signal.set()  # Signal the monitor thread to stop
    t.join()

    # montor daemon part 2
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
        'params': event['data'],
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }

# Example usage:
# event_data = { 'AES1':
#     {'length_of_message': 64,
#     'num_of_iterations': 50,
#     'num_of_processes': 8,
#     'metadata': 'Sample metadata'},
#     'index': 1

# }
# print(handler(event_data))
