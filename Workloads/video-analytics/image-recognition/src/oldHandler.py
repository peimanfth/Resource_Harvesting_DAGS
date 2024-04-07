import torchvision.models as models
from torchvision.models import SqueezeNet1_1_Weights
from torchvision import transforms
import torch
import os
import couchdb
from utils import COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD
from PIL import Image
import io
import tempfile

from multiprocessing import Pool, log_to_stderr, get_logger, Process, Queue
import logging


model = models.squeezenet1_1(weights = SqueezeNet1_1_Weights.DEFAULT)


def preprocess_image(image_bytes):
    print("inside preprocess")
    img = Image.open(io.BytesIO(image_bytes))
    img.load()  # Forces the image to be loaded from disk/memory
    print("Image loaded successfully:", img.size)
     #convert image to tensor
    img = transforms.ToTensor()(img)

    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])
    # print("transform defined")
    # img = transform(img)
    print("last line of preprocess")
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
            labels = [i for i in f]
        self.labels = labels
    def infer(self, image_name):
        print("inside infer")
        image = preprocess_image(image_name)
        #print image size
        print(f"tensor size: {image.size()}")
        self.model.eval()
        with torch.no_grad():
            out = self.model(image)
        _, indices = torch.sort(out, descending=True)
        percentages = torch.nn.functional.softmax(out, dim=1)[0] * 100
        #return the list of top 5 labels
        results = [(self.labels[idx], percentages[idx].item()) for idx in indices[0][:5]]
        return results

    def downloadImage(self, doc_name, file_name):

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
        
    
# def handler(event, context=None):
#     recog = ObjectRecognition(model, event['db_name'])
#     doc_name = event['request_id'] 
#     file_name = event['images'][event['index']] + '.jpg'
#     file = recog.downloadImage(doc_name, file_name)
#     #read tempfile as rb
#     if file:
#         result = recog.infer(file)
#         return {"prediction": result, "index": event['index']}
#     else:
#         return {"error": "Image not found or failed to download", "index": event['index']}

# if __name__ == "__main__":
#     recog = ObjectRecognition(model, 'video-bench')
#     file = recog.downloadImage('test3', 'test3-0.jpg')
#     if file:
#         result = recog.infer(file)
#         print(result)
#     else:
#         print("error: Image not found or failed to download, index: 0")
    

    

# def handle_single_image(args):
#     event, context = args
#     recog = ObjectRecognition(model, event['db_name'])
#     doc_name = event['request_id']
#     print(f"doc_name: {doc_name}")
#     # file_name = event['images'][event['index']] + '.jpg'
#     file_name = event['images'][0]
#     print(f"file_name: {file_name}")
#     image_bytes = recog.downloadImage(doc_name, file_name)
#     #print image size
#     print(f"image size: {len(image_bytes)}")
#     if image_bytes:
#         result = recog.infer(image_bytes)
#         print("result accepted")
#         return {"prediction": result, "index": 0}
#     else:
#         return {"error": "Image not found or failed to download", "index": 0}
# def torchversion(args):
#     print("inside process")
#     return torch.__version__

# def handler(event, context=None):
#     # Assuming 'event' contains a list of image indices to process
#     pool_size = 2  # Number of processes to use
#     logging.basicConfig(level=logging.INFO)
#     logger = log_to_stderr()
#     logger.setLevel(logging.INFO)
#     with Pool(pool_size) as pool:
#         tasks = [(event, context) for index in range(len(event['images']))]
#         print(tasks)
#         results = pool.map(handle_single_image, tasks)
#     return results




def handle_single_image(args, result_queue):
    try:
        event, context = args
        recog = ObjectRecognition(model, event['db_name'])
        doc_name = event['request_id']
        file_name = event['images'][event['index']] + '.jpg'
        image_bytes = recog.downloadImage(doc_name, file_name)
        if image_bytes:
            result = recog.infer(image_bytes)
            result_queue.put({"prediction": result, "index": event['index']})
        else:
            result_queue.put({"error": "Image not found or failed to download", "index": event['index']})
    except Exception as e:
        result_queue.put({"error": str(e), "index": event['index']})

def handler(event, context=None):
    processes = []
    result_queue = Queue()

    # Create a separate process for each image
    for index in range(len(event['images'])):
        p_event = event.copy()
        p_event['index'] = index  # Set the specific index for this process
        process = Process(target=handle_single_image, args=((p_event, context), result_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    # Collect all results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    return results

if __name__ == "__main__":
    dummy_event = {
    'db_name': 'video-bench',
    'request_id': 'test3',
    'images': ['test3-0', 'test3-1', 'test3-2']
    }
    print(handler(dummy_event))
