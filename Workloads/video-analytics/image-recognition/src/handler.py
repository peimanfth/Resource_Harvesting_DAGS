import torchvision.models as models
from torchvision.models import SqueezeNet1_1_Weights
from torchvision import transforms
import torch
# from torch.multiprocessing import Process, set_start_method, Queue
from multiprocessing import Process, Queue, Pool, Manager
import couchdb
from utils import COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD
from PIL import Image
import io

import queue
from threading import Thread, Event
from Monitor import monitor_peak

interval = 0.05

#Set the start method for multiprocessing
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass

# model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    return torch.unsqueeze(img, 0)

class ObjectRecognition:
    def __init__(self, model, db_name):
        self.model = model
        self.couch = couchdb.Server(COUCHDB_URL)
        self.couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
        self.db_name = db_name
        if db_name in self.couch:
            self.db = self.couch[db_name]
        else:
            self.db = self.couch.create(db_name)
        with open('imagenet_labels.txt', 'r') as f:
            self.labels = f.readlines()

    def infer(self, image_bytes):
        image = preprocess_image(image_bytes)
        self.model.eval()
        with torch.no_grad():
            out = self.model(image)
        _, indices = torch.sort(out, descending=True)
        percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
        results = [(self.labels[idx], percentages[idx].item()) for idx in indices[0][:1]]
        return results

    def download_image(self, doc_name, file_name):
        try:
            doc = self.db[doc_name]
            attachment = self.db.get_attachment(doc['_id'], file_name)
            if attachment:
                image_bytes = attachment.read()
                return image_bytes
            else:
                print(f"No attachment found for {doc_name}.")
                return None
        except couchdb.http.ResourceNotFound:
            print(f"Document {doc_name} not found.")
            return None

# def handle_single_image(args, result_queue):
#     event, index, context = args
#     model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
#     recog = ObjectRecognition(model, event['db_name'])
#     doc_name = event['request_ids'][event['index']]
#     file_name = event['images'][event['index']][index]
#     image_bytes = recog.download_image(doc_name, file_name)
#     print(f'Image downloaded with size {len(image_bytes)}')
#     if image_bytes:
#         print(f'Processing image {file_name}')
#         result = recog.infer(image_bytes)
#         result_queue.put({"prediction": result, "index": index, "image": file_name})
#     else:
#         result_queue.put({"error": "Image not found or failed to download", "index": index})
def handle_single_image(event, index, context, result_queue):
    model = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    recog = ObjectRecognition(model, event['db_name'])
    doc_name = event['request_ids'][event['index']]
    file_name = event['images'][event['index']][index]
    image_bytes = recog.download_image(doc_name, file_name)
    print(f'Image downloaded with size {len(image_bytes)}')
    if image_bytes:
        print(f'Processing image {file_name}')
        result = recog.infer(image_bytes)
        result_queue.put({"prediction": result, "index": index, "image": file_name})
    else:
        result_queue.put({"error": "Image not found or failed to download", "index": index})


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

    # result_queue = Queue()
    # processes = []
    # for index in range(len(event['images'][event['index']])):
    #     p = Process(target=handle_single_image, args=((event, index, context), result_queue))
    #     processes.append(p)
    #     p.start()

    # for p in processes:
    #     p.join()

    # results = []
    # while not result_queue.empty():
    #     results.append(result_queue.get())

    NUM_PROCESSES = (event['index'] + 1) * 3

    with Manager() as manager:
        result_queue = manager.Queue()
        with Pool(NUM_PROCESSES) as pool:
            tasks = [(event, i, context, result_queue) for i in range(len(event['images'][event['index']]))]
            pool.starmap(handle_single_image, tasks)

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())


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
        f'Recognition_{event["index"]}': results,
        'cpu_timestamp': [str(x) for x in cpu_timestamp],
        'cpu_usage': [str(x) for x in cpu_usage],
        'mem_timestamp': [str(x) for x in mem_timestamp],
        'mem_usage': [str(x) for x in mem_usage]
    }

# Main entry point for the script
if __name__ == "__main__":
    event = {'request_ids': ['58fdb1e2-8565-4b3b-a083-244da2c861e9-recog1', '58fdb1e2-8565-4b3b-a083-244da2c861e9-recog2'], 'num_frames': 2, 'db_name': 'video-bench', 'images': [[], ['0.jpg', '1.jpg']], 'index': 1}
    results = handler(event)
    print(results)
