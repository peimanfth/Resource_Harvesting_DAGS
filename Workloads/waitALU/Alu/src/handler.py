
# Copyright (c) 2020 Institution of Parallel and Distributed System, Shanghai Jiao Tong University
# ServerlessBench is licensed under the Mulan PSL v1.
# You can use this software according to the terms and conditions of the Mulan PSL v1.
# You may obtain a copy of Mulan PSL v1 at:
#     http://license.coscl.org.cn/MulanPSL
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v1 for more details.

import boto3
import time
import os
import random
from multiprocessing import Process, Pipe
from queue import Queue
from threading import Thread, Event
from Monitor import monitor_peak

defaultLoopTime = 10000000
defaultParallelIndex = 100

interval = 0.1

def handler(event, context=None):

    # monitor daemon part 1
    stop_signal = Event()
    q_cpu = Queue()
    q_mem = Queue()
    t = Thread(
        target=monitor_peak,
        args=(interval, q_cpu, q_mem),
        daemon=True
    )
    t.start()


    startTime = GetTime()
    if 'execTime' in event:
        execTime_prev = event['execTime']
    else:
        execTime_prev = 0
    if 'loopTime' in event:
        loopTime = event['loopTime']
    else:
        loopTime = defaultLoopTime
    if 'parallelIndex' in event:
        parallelIndex = event['parallelIndex']
    else:
        parallelIndex = defaultParallelIndex

    temp = alu(loopTime, parallelIndex)
    retTime = GetTime()

    # stop_signal.set()
    # t.join()
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
        "startTime": startTime,
        "retTime": retTime,
        "execTime": retTime - startTime,
        "result": temp,
        'execTime_prev': execTime_prev,
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }

def doAlu(times, childConn, clientId):
    a = random.randint(10, 100)
    b = random.randint(10, 100)
    temp = 0
    for i in range(times):
        if i % 4 == 0:
            temp = a + b
        elif i % 4 == 1:
            temp = a - b
        elif i % 4 == 2:
            temp = a * b
        else:
            temp = a / b
    print(times)
    childConn.send(temp)
    childConn.close()
    return temp

def alu(times, parallelIndex):
    per_times = int(times / parallelIndex)
    threads = []
    childConns = []
    parentConns = []
    for i in range(parallelIndex):
        parentConn, childConn = Pipe()
        parentConns.append(parentConn)
        childConns.append(childConn)
        t = Process(target=doAlu, args=(per_times, childConn, i))
        threads.append(t)
    for i in range(parallelIndex):
        threads[i].start()
    for i in range(parallelIndex):
        threads[i].join()
    
    results = []
    for i in range(parallelIndex):
        results.append(parentConns[i].recv())
    return str(results)

def GetTime():
    return int(round(time.time() * 1000))

# if __name__ == '__main__':
#     print(handler({'startTime': 1706930352494, 'retTime': 1706930352495, 'execTime': 1, 'loopTime': 10000000, 'key': '10000000'}, None))