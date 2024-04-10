import boto3
import os
import uuid

from queue import Queue
from threading import Thread, Event
from Monitor import monitor_peak

interval = 0.05

# def setEnvVar(envVar, value):
#     if envVar in os.environ:
#         print("Warning: environment variable %s is already set to %s. Overwriting it with %s" % (envVar, os.environ[envVar], value))
#     os.environ[envVar] = value


# def downloadFileFromS3(bucket_name, file_name):
#     setEnvVar('AWS_ACCESS_KEY_ID', 'AKIA26EO4UIX52Y2OUVZ')
#     setEnvVar('AWS_SECRET_ACCESS_KEY','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' )
#     s3 = boto3.client('s3')
#     s3.download_file(bucket_name, file_name, '/tmp/' + file_name)
#     return '/tmp/' + file_name

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

    request_id = str(uuid.uuid4())
    if event['video'] == 'car':
        file_name = 'car'
        doc_name = 'videos'
    elif event['video'] == 'tokyo':
        file_name = 'tokyo'
        doc_name = 'videos'
    elif event['video'] == 'mount':
        file_name = 'mount'
        doc_name = 'videos'
    elif event['video'] == 'mount1':
        file_name = 'mount1'
        doc_name = 'videos'
    elif event['video'] == 'tree':
        file_name = 'tree'
        doc_name = 'videos'
    elif event['video'] == 'snow':
        file_name = 'snow'
        doc_name = 'videos'
    
    if event['num_frames'] is not None:
        num_frames = event['num_frames']

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
        'video_name': file_name,
        'num_frames': num_frames,
        'db_name': 'video-bench',
        'doc_name': doc_name,
        'request_ids': [f'{request_id}-recog1', f'{request_id}-recog2'],
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]

    }

# if __name__ == "__main__":
#     event = {
#         "video": "car",
#         "num_frames": 2
#     }
#     print(handler(event))