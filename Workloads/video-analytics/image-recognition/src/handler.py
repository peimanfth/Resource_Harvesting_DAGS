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


model = models.squeezenet1_1(weights = SqueezeNet1_1_Weights.DEFAULT)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
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
            labels = [i for i in f]
        self.labels = labels
    def infer(self, image_name):
        image = preprocess_image(image_name)
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
    
def handler(event, context=None):
    recog = ObjectRecognition(model, event['db_name'])
    doc_name = event['request_id'] 
    file_name = event['images'][event['index']] + '.jpg'
    file = recog.downloadImage(doc_name, file_name)
    #read tempfile as rb
    if file:
        result = recog.infer(file)
        return {"prediction": result, "index": event['index']}
    else:
        return {"error": "Image not found or failed to download", "index": event['index']}

# if __name__ == "__main__":
#     recog = ObjectRecognition(model, 'video-bench')
#     file = recog.downloadImage('test2-0', 'test2-0.jpg')
#     if file:
#         result = recog.infer(file)
#         print(result)
#     else:
#         print("error: Image not found or failed to download, index: 0")
    

