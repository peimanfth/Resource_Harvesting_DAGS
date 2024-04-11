import os
import cv2
import tempfile
import couchdb
from utils import COUCHDB_PASSWORD, COUCHDB_URL, COUCHDB_USERNAME

from queue import Queue
from threading import Thread, Event


EXT = '.jpg'

class VideoDecoder:
    def __init__(self, filename, doc_name, req_ids, fanout_num, db_name):
        self.couch = couchdb.Server(COUCHDB_URL)
        self.couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
        self.db_name = db_name
        self.filename = filename
        self.doc_name = doc_name
        self.req_ids = req_ids  # List of two request IDs
        self.fanout_num = fanout_num
        if db_name in self.couch:
            self.db = self.couch[db_name]
        else:
            self.db = self.couch.create(db_name)

    def decode(self):
        try:
            doc = self.db[self.doc_name]
        except couchdb.http.ResourceNotFound:
            print(f"Document with ID {self.doc_name} not found in database {self.db_name}.")
            return []

        video_attachment = self.db.get_attachment(doc['_id'], self.filename)
        if video_attachment is None:
            print(f"Video attachment not found in document {doc['_id']}.")
            return []

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(video_attachment.read())
            temp_file.flush()

        vidcap = cv2.VideoCapture(temp_file_path)
        frames = []
        for i in range(self.fanout_num):
            success, image = vidcap.read()
            if not success:
                break
            frames.append(image)

        os.remove(temp_file_path)
        return frames

    def save_to_couchdb(self, i, frame, req_id):
        _, buffer = cv2.imencode(EXT, frame)
        frame_bytes = buffer.tobytes()
        doc_id = req_id
        try:
            doc = self.db[doc_id]
        except couchdb.http.ResourceNotFound:
            doc = {'_id': doc_id}
            self.db.save(doc)

        filename = f'{i}{EXT}'
        self.db.put_attachment(doc, frame_bytes, filename=filename, content_type='image/jpeg')
        print(f"Frame {i} uploaded to document {doc_id} successfully.")

def handler(event, context=None):


    decoder = VideoDecoder(event['video_name']+'.mp4', event['doc_name'], event['request_ids'], event['num_frames'], event['db_name'])
    frames = decoder.decode()
    if len(frames) == 2:
        num_first_doc = 1
    else:
        num_first_doc = len(frames) // 3
    images = []

    for i, frame in enumerate(frames[:num_first_doc]):
        decoder.save_to_couchdb(i, frame, event['request_ids'][0])
    images.append([f'{i}{EXT}' for i in range(num_first_doc)])

    for i, frame in enumerate(frames[num_first_doc:], start=num_first_doc):
        decoder.save_to_couchdb(i, frame, event['request_ids'][1])
    images.append([f'{i}{EXT}' for i in range(num_first_doc, len(frames))])


    return {
        'request_ids': event['request_ids'],
        'num_frames': len(frames),
        'db_name': 'video-bench',
        'images': images,
    }

if __name__ == "__main__":
    res = handler({'video_name': 'tokyo', 'num_frames': 51, 'db_name': 'video-bench', 'doc_name': 'videos', 'request_ids': ['39a9c8a2-b278-479e-b49a-bd0df702c866-recog1', '39a9c8a2-b278-479e-b49a-bd0df702c866-recog2']})
    print(res)
