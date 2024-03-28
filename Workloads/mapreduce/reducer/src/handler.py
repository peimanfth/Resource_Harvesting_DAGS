import json
import boto3
from time import time
from threading import Thread, Event
from queue import Queue
from Monitor import monitor_peak


interval = 0.02

computer_language = ["JavaScript", "Java", "PHP", "Python", "C#", "C++",
                     "Ruby", "CSS", "Objective-C", "Perl",
                     "Scala", "Haskell", "MATLAB", "Clojure", "Groovy"]


def handler(event, context=None):

    # monitor daemon part 1
    stop_signal = Event()
    q_cpu = Queue()
    q_mem = Queue()
    tmon = Thread(
        target=monitor_peak,
        args=(interval, q_cpu, q_mem, stop_signal),
        daemon=True
    )
    tmon.start()

    results = event['value']
    output = {}

    for lang in computer_language:
        output[lang] = 0

    network = 0
    reduce = 0
    start = time()
    #results is a list of dictionaries. reduce over the values of 'result' in each dictionary
    for result in results:
        for lang in computer_language:
            output[lang] += result['result'][lang]
    
    reduce += time() - start

    #results is a list of dictionaries. another key is 'mapTime' in each dictionary. Also output individual map times
    mapTimes = []
    for result in results:
        mapTimes.append(result['mapTime'])

    stop_signal.set()  # Signal the monitor thread to stop
    tmon.join()

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
        'result': output,
        'reduceTime': reduce,
        'mapperCount': len(results), #this is the number of mappers that were invoked
        'mapTimes': mapTimes,
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }
