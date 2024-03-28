import json
from threading import Thread, Event
from queue import Queue
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

    event = event["value"]
    configs_list = []
    accuracy_list = []
    for i in range(len(event)):
        for j in range(len(event[i]["trees_max_depthes"])):
            configs_list.append(event[i]["trees_max_depthes"][j])
            accuracy_list.append(event[i]["accuracies"][j])
        
    Z = [x for _, x in sorted(zip(accuracy_list, configs_list))] 
    returned_configs = Z[-10:len(accuracy_list)]
    returned_latecy = sorted(accuracy_list)[-10:len(accuracy_list)]
    print(returned_configs)
    print(returned_latecy)
    
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
        'statusCode': 200,
        'accuracy': returned_configs,
        'returned_latecy': returned_latecy,
        'all_data': json.dumps(str(event)),
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }