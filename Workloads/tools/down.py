import os
import couchdb
import boto3
import tempfile
import cv2
from utils import COUCHDB_URL, COUCHDB_DB_NAME, COUCHDB_PASSWORD, COUCHDB_USERNAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

def download_video_from_couchdb(couchdb_url, db_name, doc_id, output_path):
    couch = couchdb.Server(couchdb_url)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)

    if db_name in couch:
        db = couch[db_name]
    else:
        print(f"Database '{db_name}' not found.")
        return

    try:
        doc = db[doc_id]
    except couchdb.http.ResourceNotFound:
        print(f"Document '{doc_id}' not found in the database.")
        return

    # Specify the attachment name you want to download
    attachment_name = 'video.mp4'  # Change this to match your attachment name

    # Get the attachment content
    try:
        attachment = db.get_attachment(doc, attachment_name)
    except couchdb.http.ResourceNotFound:
        print(f"Attachment '{attachment_name}' not found in the document.")
        return

    # Write the attachment content to a file
    with open(output_path, 'wb') as f:
        f.write(attachment.read())

    print(f"Video downloaded successfully to {output_path}.")

def download_obj_from_S3(bucket_name, file_name, output_path):
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    s3 = boto3.client('s3')
    # tmp = tempfile.NamedTemporaryFile(suffix='.mp4')
        #download from s3
    s3.download_file(bucket_name, file_name, output_path)
    #save tmp to output_path


if __name__ == "__main__":
    couchdb_url = COUCHDB_URL
    db_name = COUCHDB_DB_NAME
    document_id = 'cars'  # Replace with your document ID
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = 'videos/downcars.mp4'
    video_path = os.path.join(current_dir, relative_path)

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_file_path = temp_file.name
    download_obj_from_S3('video-bench', 'car.mp4', temp_file_path)

    vidcap = cv2.VideoCapture(temp_file_path)
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frames.append(image)  # Append the raw image
    print(len(frames))
