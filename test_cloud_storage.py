from os import environ
import os
from google.cloud import storage
from cloud_storage import Cloud_storage
from os import listdir
from os.path import isfile, join
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/tmt/Documents/face_recognition/my_app/optical-carrier-262904-5a9b32e70b49.json'
# Google Cloud Storage
# bucketName = environ.get('tung-recognition')
# bucketFolder = environ.get('recognize')

# # Data
# localFolder = environ.get('/home/tmt/Documents/face_recognition/my_app/storage/trained/Nancy')
# print(localFolder)
# # print(os.environ)
# storage_client = storage.Client()
# bucket = storage_client.get_bucket('tung-recognition')

# destination_blob_name = 'recognize/'
# blob = bucket.blob(destination_blob_name)
# source_file_name = '/home/tmt/Documents/face_recognition/my_app/storage/unknown/Nancy/nc2-678x381.jpg'
# blob.upload_from_filename(source_file_name)

# print('File {} uploaded to {}.'.format(
#     source_file_name,
#     destination_blob_name))


def download_blob(bucket_name, buket_folder, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(buket_folder+'/'+source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


# buket_folder = 'recognize/trained/Tung/'
# upload_blob('tung-recognition', '/home/tmt/Documents/face_recognition/my_app/storage/trained/Tung/1578040709beauty_20190915092027.jpg', 'tung.jpg')

storage_client = storage.Client()
bucket = storage_client.bucket('tung-recognition')
# localFolder = '/home/tmt/Documents/face_recognition/my_app/storage/trained/Tung/'
# bucket.delete(force=True)
# storage_client.create_bucket('tung-recognition')
# files = [f for f in listdir(localFolder) if isfile(join(localFolder, f))]
# for file in files:
#     localFile = localFolder + file
#     blob = bucket.blob(bucketFolder + file)
#     blob.upload_from_filename(localFile)
#     print("Upload file {}".format(localFile))
# download_blob('tung-recognition', 'tung.jpg', '/home/tmt/Documents/face_recognition/my_app/cloud.jpg')
# cloud = Cloud_storage()
# path = '/home/tmt/Documents/face_recognition/my_app/storage/trained/Tung/1578040709beauty_20190915092027.jpg'
# destination_blob_name = 'tung2.jpg'
# cloud.upload_blob(path, destination_blob_name)


def create_folder(bucket_name, destination_folder_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_folder_name)

    blob.upload_from_string('')

    print('Created {} .'.format(
        destination_folder_name))


# bucket_name = 'tung-recognition'
# storage_client = storage.Client()
# folder = create_folder(bucket_name, 'recognize/trained/Tung/')
# bucket = storage_client.get_bucket(bucket_name)
# if bucket != None:
#     print("YES")
# if storage_client.get_bucket(bucket_name) != None:
#     storage_client.get_bucket(bucket_name).delete()
# else:
#     bucket = storage_client.create_bucket(bucket_name)
#     print("Bucket {} created".format(bucket.name))


def get_all_file():
    directory = '/home/tmt/Documents/face_recognition/my_app/storage/trained'
    # dir = os.walk(directory)
    # for r, d, f in os.walk(directory):
    #     print(r)
    #     for file in f:
    #         print(file)
    # print(dir)

    files = os.listdir(directory)
    for name in files:
        print(name)
        sub_dir = os.listdir(directory+'/'+name)
        for i in sub_dir:
            print(i)


def upload_blob(bucket_name, bucket_folder, source_file_name, destination_blob_name,bucket):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(bucket_folder+'/'+destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def run():
    bucket_name = 'tung-recognition'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    path = '/home/tmt/Documents/face_recognition/my_app/storage/trained'
    files = os.listdir(path)
    for name in files:

        # folder = create_folder(bucket_name, 'recognize/trained/'+name+'/')
        bucket_folder = 'recognize/trained/'+name
        sub_dir = os.listdir(path+'/'+name)
        for image_name in sub_dir:
            image_path = path+'/'+name+'/'+image_name
            print(image_path)
            try:
                upload_blob(bucket_name, bucket_folder, image_path, image_name,bucket)
            except:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                upload_blob(bucket_name, bucket_folder, image_path, image_name,bucket)


run()
# get_all_file()
