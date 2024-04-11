from time import time
import random
import string
import pyaes
from multiprocessing import Process, Manager, cpu_count
from queue import Queue
from threading import Thread, Event
from Monitor import monitor_peak

interval = 0.05

def generate(length):
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def encrypt_decrypt_iterations(start, end, message, KEY, final_ciphertexts, final_plaintexts, process_index):
    print(f"Process {process_index} started at {time()}")  # Log when the process starts
    local_ciphertext = ""
    local_plaintext = message
    
    for loops in range(start, end):
        aes_encrypt = pyaes.AESModeOfOperationCTR(KEY)
        local_ciphertext = aes_encrypt.encrypt(local_plaintext)
        
        aes_decrypt = pyaes.AESModeOfOperationCTR(KEY)
        local_plaintext = aes_decrypt.decrypt(local_ciphertext)
    
    # Only store the final ciphertext and plaintext for this process
    final_ciphertexts[process_index] = local_ciphertext.hex()  # Store as hex string
    final_plaintexts[process_index] = local_plaintext.decode('utf-8')  # Decode to string
    print(f"Process {process_index} ended at {time()}")  # Log when the process ends

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

    latencies = {}
    timestamps = {}
    print(event['params'])
    event = event['params'][f'AES{event["index"]}']
    timestamps["starting_time"] = time()
    length_of_message = event['length_of_message']
    num_of_iterations = event['num_of_iterations']
    num_of_processes = event.get('num_of_processes', cpu_count())  # Use CPU count if not specified
    
    print(num_of_processes)

    KEY = b'\xa1\xf6%\x8c\x87}_\xcd\x89dHE8\xbf\xc9,'

    manager = Manager()
    final_ciphertexts = manager.dict()  # Shared dict to store final ciphertexts
    final_plaintexts = manager.dict()  # Shared dict to store final plaintexts

    # Determine the range of iterations for each process
    iterations_per_process = num_of_iterations // num_of_processes
    processes = []

    start = time()
    for i in range(num_of_processes):
        # Generate a unique message for each process
        unique_message = generate(length_of_message)
        
        start_iter = i * iterations_per_process
        # Ensure the last process handles any remaining iterations
        end_iter = start_iter + iterations_per_process if i != num_of_processes - 1 else num_of_iterations
        process = Process(target=encrypt_decrypt_iterations, args=(start_iter, end_iter, unique_message, KEY, final_ciphertexts, final_plaintexts, i))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    latency = time() - start
    latencies["function_execution"] = latency
    timestamps["finishing_time"] = time()

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
        "latencies": latencies,
        "timestamps": timestamps,
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
