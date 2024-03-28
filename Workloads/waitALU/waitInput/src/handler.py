
# Copyright (c) 2020 Institution of Parallel and Distributed System, Shanghai Jiao Tong University
# ServerlessBench is licensed under the Mulan PSL v1.
# You can use this software according to the terms and conditions of the Mulan PSL v1.
# You may obtain a copy of Mulan PSL v1 at:
#     http://license.coscl.org.cn/MulanPSL
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v1 for more details.

import time
import os
from queue import Queue
from threading import Thread, Event
from Monitor import monitor_peak

defaultKey = "10000000"

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
    if 'key' in event:
        key = event['key']
    else:
        key = defaultKey

    saveFile(key)
    loopTime = extractLoopTime(key)
    # meta = extractMetadata(key)
    # upload_file(key)
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
        "loopTime": loopTime,
        "key": key,
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }


def saveFile(key):
    filepath = "/tmp/%s" %key

    # write key to file
    txtfile = open(filepath, 'w')
    txtfile.write(key)
    txtfile.close()


def extractLoopTime(key):
    filepath = "/tmp/%s" %key
    txtfile = open(filepath, 'r')
    loopTime = int(txtfile.readline())
    print("loopTime: " + str(loopTime))
    txtfile.close()
    return loopTime


def GetTime():
    return int(round(time.time() * 1000))

# if __name__ == '__main__':
#     print(hadndler({"key": "10000000465"}, None))