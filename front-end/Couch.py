import os
import time
import couchdb
from utils import run_cmd

class Couch:
    def __init__(self, couchdb_url, username, password):
        self.couchdb_url = couchdb_url
        self.username = username
        self.password = password
        self.couch = couchdb.Server(couchdb_url)
        self.couch.resource.credentials = (username, password)

    def upload_file_to_couchdb(self, file_path, db_name, doc_id, content_type, filename):
        if db_name in self.couch:
            db = self.couch[db_name]
        else:
            db = self.couch.create(db_name)

        with open(file_path, 'rb') as f:
            file_data = f.read()

        try:
            doc = db[doc_id]
        except couchdb.http.ResourceNotFound:
            doc = {'_id': doc_id}
            db.save(doc)  # Save the new document to the database

        db.put_attachment(doc, file_data, filename=filename, content_type=content_type)
        print(f"{filename} uploaded successfully.")

    def poll_couchdb_for_results(self, db_name, doc_id, timeout=240):
        if db_name in self.couch:
            db = self.couch[db_name]
        else:
            print(f"Database {db_name} not found.")
            return None

        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                # db = self.couch[db_name]
                doc = db[doc_id]
                if doc:
                    print("Result found in CouchDB.")
                    return doc
                time.sleep(5)  # Wait for a short period before trying again
            except Exception as e:
                # print(f"Error polling CouchDB: {e}")
                pass
                
        print("Timeout reached without finding results.")
        #print all docs in the database
        # db = self.couch[db_name]
        return None

    def get_doc_with_name(self, db_name, doc_name):
        if db_name in self.couch:
            db = self.couch[db_name]
        else:
            print(f"Database {db_name} not found.")
            return None

        for doc_id in db:
            doc = db[doc_id]
            if doc.get('name') == doc_name:
                    return doc

        return None

    def wipe_couchdb(self):
        run_cmd("../ansible/wipe.sh")


    def reset_couchdb_database(self, db_name, recreate=True):
        """
        Completely delete a CouchDB database and optionally recreate it.

        Parameters:
        - db_url: URL to the CouchDB instance.
        - db_name: Name of the database to delete.
        - recreate: Boolean indicating whether to recreate the database after deletion. Defaults to True.
        """

        # Delete the database
        try:
            del self.couch[db_name]
            print(f"Database '{db_name}' deleted successfully.")
        except couchdb.http.ResourceNotFound:
            print(f"Database '{db_name}' not found.")

        # Recreate the database if requested
        if recreate:
            self.couch.create(db_name)
            print(f"Database '{db_name}' created successfully.")
    
    def delete_guest_docs(self, db_name="whisk_local_activations"):

        # Access the database
        try:
            db = self.couch[db_name]
        except couchdb.http.ResourceNotFound:
            print(f"Database {db_name} not found.")
            return

        # Fetch all document IDs that start with "guest"
        guest_docs = [doc_id for doc_id in db if doc_id.startswith('guest')]

        # Delete each document
        for doc_id in guest_docs:
            doc = db[doc_id]  # Get the document to obtain its _rev
            db.delete(doc)  # Delete the document
            print(f"Deleted document: {doc_id}")
    
    def delete_docs_except_provided(self, db_name, exclude_doc_ids):
        """
        Deletes all documents in the specified database except for those whose IDs are provided in the exclude_doc_ids list.

        Parameters:
        - db_name: Name of the database from which documents are to be deleted.
        - exclude_doc_ids: List of document IDs that should not be deleted.
        """
        try:
            db = self.couch[db_name]
        except couchdb.http.ResourceNotFound:
            print(f"Database {db_name} not found.")
            return

        # Fetch all document IDs and revisions
        docs_to_delete = []
        for doc_id in db:
            if doc_id not in exclude_doc_ids:
                doc = db[doc_id]
                docs_to_delete.append({"_id": doc_id, "_rev": doc["_rev"], "_deleted": True})

        # Bulk delete
        if docs_to_delete:
            response = db.update(docs_to_delete)
            for success, doc_id, rev_or_exc in response:
                if success:
                    print(f"Deleted document: {doc_id}")
                else:
                    print(f"Failed to delete document: {doc_id}, Error: {rev_or_exc}")
        else:
            print("No documents to delete.")


# Example usage
if __name__ == "__main__":
    from params import COUCHDB_URL, VA_DB_NAME, COUCHDB_PASSWORD, COUCHDB_USERNAME, MR_DB_NAME, ML_DB_NAME
    
    couch = Couch(COUCHDB_URL, COUCHDB_USERNAME, COUCHDB_PASSWORD)

    # Upload a video file example
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/car.mp4')
    couch.upload_file_to_couchdb(video_path, VA_DB_NAME, 'videos', 'video/mp4', 'car.mp4')

    # Upload a Wikipedia XML file
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/enwiki-latest-pages-articles17.xml')
    couch.upload_file_to_couchdb(xml_path, MR_DB_NAME, 'wikipedia_data', 'application/xml', 'wikipedia_data.xml')

    # Upload a digits text file
    digits_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/Digits_Train.txt')
    couch.upload_file_to_couchdb(digits_path, ML_DB_NAME, 'train_digits', 'text/plain', 'digits.txt')

    # Poll for results example
    # result_doc = couch.poll_couchdb_for_results(VA_DB_NAME, 'videos')
    # if result_doc:
    #     print("Document found:", result_doc)
