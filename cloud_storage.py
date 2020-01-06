from os import environ
import os
from google.cloud import storage

from os import path, getcwd
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/tmt/Documents/face_recognition/my_app/optical-carrier-262904-5a9b32e70b49.json'


class Cloud_storage:
    def __init__(self):
        # self.trained=app.config['trained']
        # self.unknown=app.config['unknown']
        self.trained = path.join(getcwd(), 'storage', 'trained')
        self.unknown = path.join(getcwd(), 'storage', 'trained')
        
        self.bucket_name = 'tung-recognition'
        self.bucketFolder = 'recognize'
        storage_client = storage.Client()
        self.bucket = storage_client.bucket(self.bucket_name)

    def download_blob(self, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        # bucket_name = "your-bucket-name"
        # source_blob_name = "storage-object-name"
        # destination_file_name = "local/path/to/file"

        # storage_client = storage.Client()

        # bucket = storage_client.bucket(self.bucket_name)

        blob = self.bucket.blob(self.bucketFolder+'/'+source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(
            "Blob {} downloaded to {}.".format(
                source_blob_name, destination_file_name
            )
        )

    def upload_blob(self, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # bucket_name = "your-bucket-name"
        # source_file_name = "local/path/to/file"
        # destination_blob_name = "storage-object-name"

        # storage_client = storage.Client()
        # bucket = storage_client.bucket(self.bucket_name)
        blob = self.bucket.blob(self.bucketFolder+'/'+'tung/'+destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )
