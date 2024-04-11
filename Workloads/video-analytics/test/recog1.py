import torchvision.models as models
from torchvision.models import SqueezeNet1_1_Weights
from torchvision import transforms
import torch
from multiprocessing import Process, Queue, Pool, Manager
import couchdb
from utils import COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD
from PIL import Image
import io
import time
from threading import Thread, Event

# Define number of processes
NUM_PROCESSES = 6

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
        results = [(self.labels[idx], percentages[idx].item()) for idx in indices[0][:5]]
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

def handle_images(events, context, result_queue):
    for event, index in events:
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

def handler(event, context=None, num_processes=6):
    with Manager() as manager:
        result_queue = manager.Queue()
        processes = []
        task_list = [(event, i) for i in range(len(event['images'][event['index']]))]

        # Distributing tasks among the fixed number of processes
        tasks_per_process = len(task_list) // num_processes
        for i in range(num_processes):
            start_index = i * tasks_per_process
            if i == num_processes - 1:  # Ensure last process gets all remaining tasks
                end_index = len(task_list)
            else:
                end_index = start_index + tasks_per_process
            process_task_list = task_list[start_index:end_index]
            p = Process(target=handle_images, args=(process_task_list, context, result_queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        return {
            f'Recognition_{event["index"]}': results
        }
if __name__ == "__main__":
    start_time = time.time()
    event = {'request_ids': ['39a9c8a2-b278-479e-b49a-bd0df702c866-recog1', '39a9c8a2-b278-479e-b49a-bd0df702c866-recog2'], 'num_frames': 51, 'db_name': 'video-bench', 'images': [['0.jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg', '16.jpg'], ['17.jpg', '18.jpg', '19.jpg', '20.jpg', '21.jpg', '22.jpg', '23.jpg', '24.jpg', '25.jpg', '26.jpg', '27.jpg', '28.jpg', '29.jpg', '30.jpg', '31.jpg', '32.jpg', '33.jpg', '34.jpg', '35.jpg', '36.jpg', '37.jpg', '38.jpg', '39.jpg', '40.jpg', '41.jpg', '42.jpg', '43.jpg', '44.jpg', '45.jpg', '46.jpg', '47.jpg', '48.jpg', '49.jpg', '50.jpg']], 'index': 1}
    results = handler(event)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    print(results)
    with open('results.json', 'w') as f:
        f.write(str(results))
