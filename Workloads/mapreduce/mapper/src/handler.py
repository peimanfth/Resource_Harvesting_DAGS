import json
import couchdb
from time import time
from params import *
from threading import Thread, Event
from queue import Queue
from Monitor import monitor_peak

# Create S3 session
# s3_client = boto3.client('s3', aws_access_key_id=accessKeyId, aws_secret_access_key=accessKey)

interval = 0.02

subs = "</title><text>"
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

    # job_bucket = jobBucket
    doc = event['requestId']
    src_keys = event['keys']
    mapper_id = event['mapperId']

    couch = couchdb.Server(COUCHDB_URL)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    db = couch[MR_DB_NAME]

    output = {}

    for lang in computer_language:
        output[lang] = 0

    network = 0
    map = 0
    # keys = src_keys.split('/')
    key = src_keys[mapper_id]
    
    print(key)
    start = time()
    # response = s3_client.get_object(Bucket=src_bucket, Key=key)
    response = db.get_attachment(doc, key)
    contents = response.read().decode('utf-8')
    # print(contents)
    network += time() - start

    start = time()
    for line in contents.split('\n')[:-1]:
        idx = line.find(subs)
        text = line[idx + len(subs): len(line) - 16]
        for lang in computer_language:
            if lang in text:
                output[lang] += 1

    map += time() - start

    print(output)

    # metadata = {
    #     'output': str(output),
    #     'network': str(network),
    #     'map': str(map)
    # }

    # start = time()
    # s3_client.Bucket(job_bucket).put_object(Key=str(mapper_id), Body=json.dumps(output), Metadata=metadata)
    # network += time() - start
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
        'mapTime': map,
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }

# if __name__ == '__main__':
#     event = {'numPartitions': 3, 'keys': ['partition_88b82b26-7e40-4961-981e-a2bd4fa17cf8_1.xml', 'partition_88b82b26-7e40-4961-981e-a2bd4fa17cf8_2.xml', 'partition_88b82b26-7e40-4961-981e-a2bd4fa17cf8_3.xml'], 'requestId': '88b82b26-7e40-4961-981e-a2bd4fa17cf8', 'mapperId':0}
#     print(handler(event))
