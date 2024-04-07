import os
import couchdb
import requests
from utils import COUCHDB_URL, VA_DB_NAME, COUCHDB_PASSWORD, COUCHDB_USERNAME, MR_DB_NAME, ML_DB_NAME

def upload_video_to_couchdb(video_path, couchdb_url, db_name, doc_id):
    couch = couchdb.Server(couchdb_url)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    
    if db_name in couch:
        db = couch[db_name]
    else:
        db = couch.create(db_name)

    # Read the MP4 file as binary data
    with open(video_path, 'rb') as f:
        file_data = f.read()

    # Set the file content type (in this case, 'video/mp4')
    headers = {'Content-Type': 'video/mp4'}

    # Upload the file to CouchDB
    try:
        doc = db[doc_id]
    except couchdb.http.ResourceNotFound:
        doc = {'_id': doc_id}
        db.save(doc)  # Save the new document to the database

    # Attach the file to the document
    db.put_attachment(doc, file_data, filename='car.mp4', content_type='video/mp4')

    print("MP4 file uploaded successfully.")

def upload_wikipedia_xml_to_couchdb(xml_path, couchdb_url, db_name, doc_id):
    couch = couchdb.Server(couchdb_url)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    
    if db_name in couch:
        db = couch[db_name]
    else:
        db = couch.create(db_name)

    # Read the XML file as binary data
    with open(xml_path, 'rb') as f:
        file_data = f.read()

    # Set the file content type (in this case, 'application/xml')
    headers = {'Content-Type': 'application/xml'}

    # Upload the file to CouchDB
    try:
        doc = db[doc_id]
    except couchdb.http.ResourceNotFound:
        doc = {'_id': doc_id}
        db.save(doc)  # Save the new document to the database

    # Attach the file to the document
    db.put_attachment(doc, file_data, filename='wikipedia_data.xml', content_type='application/xml')

    print("XML file uploaded successfully.")


def upload_digits_to_couchdb(digits_path, couchdb_url, db_name, doc_id):
    couch = couchdb.Server(couchdb_url)
    couch.resource.credentials = (COUCHDB_USERNAME, COUCHDB_PASSWORD)
    
    if db_name in couch:
        db = couch[db_name]
    else:
        db = couch.create(db_name)

    # Read the digits file as binary data
    with open(digits_path, 'rb') as f:
        file_data = f.read()

    # Set the file content type (in this case, 'text/plain')
    headers = {'Content-Type': 'text/plain'}

    # Upload the file to CouchDB
    try:
        doc = db[doc_id]
    except couchdb.http.ResourceNotFound:
        doc = {'_id': doc_id}
        db.save(doc)  # Save the new document to the database

    # Attach the file to the document
    db.put_attachment(doc, file_data, filename='digits.txt', content_type='text/plain')

    print("Digits file uploaded successfully.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = 'data/car.mp4'
    video_path = os.path.join(current_dir, relative_path)
    couchdb_url = COUCHDB_URL
    db_name = VA_DB_NAME
    document_id = 'videos'

    upload_video_to_couchdb(video_path, couchdb_url, db_name, document_id)


    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = 'data/enwiki-latest-pages-articles17.xml'  # Update this path to your XML file's location
    xml_path = os.path.join(current_dir, relative_path)
    couchdb_url = COUCHDB_URL
    db_name = MR_DB_NAME
    document_id = 'wikipedia_data'  # You might want to choose a different document ID

    upload_wikipedia_xml_to_couchdb(xml_path, couchdb_url, db_name, document_id)


    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = 'data/Digits_Test.txt'
    rel_train_path = 'data/Digits_Train.txt'
    train_path = os.path.join(current_dir, rel_train_path)
    couchdb_url = COUCHDB_URL
    db_name = ML_DB_NAME
    document_id = 'train_digits'

    upload_digits_to_couchdb(train_path, couchdb_url, db_name, document_id)
    
