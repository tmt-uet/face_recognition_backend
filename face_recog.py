import faiss
from numpy import load
import face_recognition
from os import path
import numpy as np
from flask import Flask, json
from numpy import save
from numpy import load


class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.model = app.config['model']
        self.db = app.db
        self.faces = []  # storage all faces in caches array of face object
        self.known_encoding_faces = []  # faces data for recognition
        self.face_user_keys = {}
        self.known_face_names = []
        self.known_encoding_faces2 = []
        # self.load_all()
        self.gpu_index = None
        self.cpu_index = None
        # self.init_index()

    def init_index(self):
        d = 128
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)

        self.cpu_index = faiss.IndexFlatL2(d)
        self.cpu_index.add(self.known_encoding_faces2)

        # self.gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        #     self.cpu_index
        # )

        # self.gpu_index.add(self.known_encoding_faces2)              # add vectors to the index
        # print('index', self.gpu_index.ntotal)

    def load_user_by_index_key(self, index_key=0):

        key_str = str(index_key)

        if key_str in self.face_user_keys:
            return self.face_user_keys[key_str]

        return None

    def load_train_file_by_name(self, name, filename):
        trained_storage = path.join(self.storage, 'trained', name)
        return path.join(trained_storage, filename)

    def load_unknown_file_by_name(self, filename):
        unknown_storage = path.join(self.storage, 'unknown')
        return path.join(unknown_storage, filename)

    def load_all(self):
        print("First Work")
        # results = self.db.select('SELECT faces.id, faces.user_id, faces.filename, faces.created FROM faces')
        results = self.db.select(
            'SELECT faces.id, faces.user_id, faces.filename,faces.created, users.id, users.name, users.created  FROM users LEFT JOIN faces ON faces.user_id = users.id')
        for row in results:
            id = row[0]
            user_id = row[1]
            filename = row[2]
            created = row[3]
            name = row[5]
            # face = {
            #     "id": row[0],
            #     "user_id": user_id,
            #     "filename": filename,
            #     "created": created
            # }
            # self.faces.append(face)
            print('name ', name)
            print('filename', filename)
            self.known_face_names.append(name)

            # face_image = face_recognition.load_image_file(self.load_train_file_by_name(name, filename))
            # face_image_encoding = face_recognition.face_encodings(face_image)[0]

        #     path_np = self.load_train_file_by_name(name, filename)+str('.npy')
        #     face_image_encoding = load(path_np)

        #     index_key = len(self.known_encoding_faces)

        #     self.known_encoding_faces.append(face_image_encoding)

        #     index_key_string = str(index_key)
        #     self.face_user_keys[index_key_string] = user_id

        # self.known_encoding_faces2 = self.known_encoding_faces
        # self.known_encoding_faces2 = np.asarray(self.known_encoding_faces2)
        # self.known_encoding_faces2 = np.reshape(self.known_encoding_faces2, (len(self.known_encoding_faces), 128))
        # self.known_encoding_faces2 = self.known_encoding_faces2.astype(np.float32)
        # print(self.known_encoding_faces2.shape)
        self.known_encoding_faces2 = np.load(path.join(self.model, 'model.npy'))
        print(self.known_encoding_faces2.shape)
        print('load model done')

    def update_model(self):
        print("First Work")
        # results = self.db.select('SELECT faces.id, faces.user_id, faces.filename, faces.created FROM faces')
        results = self.db.select(
            'SELECT faces.id, faces.user_id, faces.filename,faces.created, users.id, users.name, users.created  FROM users LEFT JOIN faces ON faces.user_id = users.id')
        for row in results:
            id = row[0]
            user_id = row[1]
            filename = row[2]
            created = row[3]
            name = row[5]
            # face = {
            #     "id": row[0],
            #     "user_id": user_id,
            #     "filename": filename,
            #     "created": created
            # }
            # self.faces.append(face)
            # print('name ', name)
            # print('filename', filename)
            # self.known_face_names.append(name)

            # face_image = face_recognition.load_image_file(self.load_train_file_by_name(name, filename))
            # face_image_encoding = face_recognition.face_encodings(face_image)[0]

            path_np = self.load_train_file_by_name(name, filename)+str('.npy')
            face_image_encoding = load(path_np)

            index_key = len(self.known_encoding_faces)

            self.known_encoding_faces.append(face_image_encoding)

            index_key_string = str(index_key)
            self.face_user_keys[index_key_string] = user_id

        self.known_encoding_faces2 = self.known_encoding_faces
        self.known_encoding_faces2 = np.asarray(self.known_encoding_faces2)
        self.known_encoding_faces2 = np.reshape(self.known_encoding_faces2, (len(self.known_encoding_faces), 128))
        self.known_encoding_faces2 = self.known_encoding_faces2.astype(np.float32)

        path_model = path.join(self.model, 'model.npy')

        save(path_model, self.known_encoding_faces2)
        print("saved model done")

        # print(row)

    def recognize2(self, name, unknown_face_location, unknown_face_image):
        output = {}

        unknown_encoding = face_recognition.face_encodings(unknown_face_image, known_face_locations=unknown_face_location)[0]
        unknown_encoding_faiss = np.reshape(unknown_encoding, (1, 128))
        unknown_encoding_faiss = unknown_encoding_faiss.astype(np.float32)
        k = 2
        # D, I = self.gpu_index.search(unknown_encoding_faiss, k)
        D, I = self.cpu_index.search(unknown_encoding_faiss, k)
        best_match_index = I[0][0]
        print('best_match_index', best_match_index)
        unknow_name = self.known_face_names[best_match_index]
        print('name', unknow_name)
        if str(unknow_name) == str(name):

            output['name'] = unknow_name
            output['message'] = 'Khuôn mặt này  hợp lệ'
            # output['face_distance_average'] = face_distance_average
            output['code'] = 1
            output['status'] = 'VALID'
        else:
            output['message'] = 'Khuôn mặt này không hợp lệ'
            # output['face_distance_average'] = face_distance_average
            output['code'] = 2
            output['status'] = 'INVALID'

        return output
