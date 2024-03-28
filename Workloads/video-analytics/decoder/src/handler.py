import os
import cv2
import tempfile
import couchdb
from utils import COUCHDB_PASSWORD, COUCHDB_URL, COUCHDB_USERNAME


EXT = '.jpg'  # Replace this with the appropriate file extension
class VideoDecoder:
    def __init__(self, filename,doc_name, req_id, fanout_num, db_name):
        self.couch = couchdb.Server(COUCHDB_URL)
        self.couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
        self.db_name = db_name
        self.filename = filename
        self.doc_name = doc_name
        self.req_id = req_id
        self.fanout_num = fanout_num
        if db_name in self.couch:
            self.db = self.couch[db_name]
        else:
            self.db = self.couch.create(db_name)


    def decode(self):
        # with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        #     temp_file_path = temp_file.name
        # self.s3.download_file(self.bucket_name, self.filename, temp_file_path)

        # vidcap = cv2.VideoCapture(temp_file_path)
        # frames = []
        # for i in range(self.fanout_num):
        #     _, image = vidcap.read()
        #     frames.append(image)
        # print(len(frames))
        # return frames
        # # while True:
        # #     success, image = vidcap.read()
        # #     if not success:
        # #         break
        # #     frames.append(image)  # Append the raw image frames
        # # print(len(frames))
        # # return frames

        try:
            doc = self.db[self.doc_name]
        except couchdb.http.ResourceNotFound:
            print(f"Document with ID {self.doc_name} not found in database {self.db_name}.")
            return []

        # Download the video attachment into a temporary file
        video_attachment = self.db.get_attachment(doc['_id'], self.filename)  # Assuming the attachment is named 'video.mp4'
        if video_attachment is None:
            print(f"Video attachment not found in document {doc['_id']}.")
            return []

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(video_attachment.read())
            temp_file.flush()

        # Use OpenCV to decode the video
        vidcap = cv2.VideoCapture(temp_file_path)
        frames = []
        for i in range(self.fanout_num):
            success, image = vidcap.read()
            if not success:
                break  # Exit loop if no frame is read
            frames.append(image)

        os.remove(temp_file_path)  # Clean up the temporary file
        return frames

    # def upload(self, i, frame_bytes):
    #     filename = f'{self.req_id}-{i}{EXT}'
    #     self.s3.put_object(Bucket=self.bucket_name, Key=filename, Body=frame_bytes)
    #     return filename
    # def saveLocally(self, i, frame):
    #     filename = f'{self.req_id}-{i}{EXT}'
    #     cv2.imwrite(filename, frame)

    # def saveS3(self, i, frame):
    #     with tempfile.NamedTemporaryFile(suffix=EXT, delete=False) as temp_file:
    #         temp_file_path = temp_file.name
    #     cv2.imwrite(temp_file_path, frame)
    #     self.s3.upload_file(temp_file_path, self.bucket_name, f'{self.req_id}-{i}{EXT}')
    #     os.remove(temp_file_path)

    def save_to_couchdb(self, i, frame):
        # Convert frame to JPG bytes
        _, buffer = cv2.imencode(EXT, frame)
        frame_bytes = buffer.tobytes()

        doc_id = f'{self.req_id}'
        try:
            doc = self.db[doc_id]
        except couchdb.http.ResourceNotFound:
            doc = {'_id': doc_id}
            self.db.save(doc)

        # Attach the frame to the document
        self.db.put_attachment(doc, frame_bytes, filename=f'{doc_id}-{i}{EXT}', content_type='image/jpeg')
        print(f"Frame {i} uploaded successfully.")

def handler(event, context=None):
    """Decoder handler"""
    decoder = VideoDecoder(event['video_name']+'.mp4', event['doc_name'], event['request_id'], event['num_frames'], event['db_name'])
    frames = decoder.decode()

    for i, frame in enumerate(frames):
        decoder.save_to_couchdb(i, frame)
    return {
        'request_id': event['request_id'],
        'num_frames': event['num_frames'],
        'db_name': 'video-bench',
        'images': [f'{event["request_id"]}-{i}' for i in range(event['num_frames'])]
    
    }

# if __name__ == "__main__":
#     res = handler({
#         'video_name': 'car',
#         'doc_name': 'videos',
#         'num_frames': 3,
#         'request_id': 'test3',
#         'db_name': 'video-bench'
#     })
#     print(type(res))
