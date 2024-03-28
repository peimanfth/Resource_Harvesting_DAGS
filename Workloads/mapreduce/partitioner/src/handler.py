import math
import couchdb
import uuid
from params import COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD, MR_DB_NAME, COUCHDB_DOC
from threading import Thread, Event
from queue import Queue
from Monitor import monitor_peak


interval = 0.02
def calculate_partition_sizes(total_lines, num_partitions, scaling_factor):
    """
    Calculate sizes of each partition according to a scaling factor.
    Returns a list of integers representing the number of lines each partition should have.
    """
    divisor = sum(scaling_factor ** i for i in range(num_partitions))
    initial_partition_size = total_lines / divisor
    partition_sizes = [math.ceil(initial_partition_size * (scaling_factor ** i)) for i in range(num_partitions)]
    
    return partition_sizes

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

    couch = couchdb.Server(COUCHDB_URL)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    db = couch[MR_DB_NAME]
    doc = db[COUCHDB_DOC]
    response = db.get_attachment(doc['_id'], event['data'] + '.xml')

    request_id = uuid.uuid4()
    print('Partitioning...')

    content = response.read().decode('utf-8')
    total_lines = content.count('</text>')
    num_partitions = event['numPartitions']
    scaling_factor = 2  # This can be adjusted as needed

    partition_sizes = calculate_partition_sizes(total_lines, num_partitions, scaling_factor)

    records = content.split('</text>')
    current_partition_index = 0
    current_lines = 0
    current_content = ''
    all_partitions = []

    try:
        doc = db[str(request_id)]
    except couchdb.http.ResourceNotFound:
        doc = {'_id': str(request_id)}
        db.save(doc)

    for record in records[:-1]:  # Exclude the last empty record if it exists
        current_content += record + '</text>'
        current_lines += 1

        # Check if current partition reached its limit
        if current_lines >= partition_sizes[current_partition_index]:
            output_key = f'partition_{request_id}_{current_partition_index + 1}.xml'
            output_bytes = current_content.encode('utf-8')
            db.put_attachment(doc, output_bytes, filename=output_key, content_type='application/xml')
            
            # Reset for next partition
            current_content = ''
            current_lines = 0
            current_partition_index += 1
            all_partitions.append(output_key)

            if current_partition_index >= num_partitions:
                break  # Stop if we have created all partitions

    # Handle any remaining content for the last partition
    if current_content:
        output_key = f'partition_{request_id}_{current_partition_index + 1}.xml'
        output_bytes = current_content.encode('utf-8')
        db.put_attachment(doc, output_bytes, filename=output_key, content_type='application/xml')
        all_partitions.append(output_key)

    print('Partitioned successfully')

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
        'numPartitions': num_partitions,
        'keys': all_partitions,
        'requestId': str(request_id),
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }

# if __name__ == '__main__':
#     print(handler({'numPartitions': 3, 'data': 'wikipedia_data'}))
