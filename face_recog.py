import face_recognition
from os import path
import numpy as np
/from flask import Flask, json
from numpy import load
import faiss


class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.db = app.db
        self.faces = []  # storage all faces in caches array of face object
        self.known_encoding_faces = []  # faces data for recognition
        self.face_user_keys = {}
        self.known_face_names = []
        self.known_encoding_faces2 = []
        self.load_all()
        self.gpu_index = None
        self.cpu_index = None
        self.init_index()

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
            face = {
                "id": row[0],
                "user_id": user_id,
                "filename": filename,
                "created": created
            }
            self.faces.append(face)
            print('name ', name)
            print('filename', filename)
            self.known_face_names.append(name)

            face_image = face_recognition.load_image_file(self.load_train_file_by_name(name, filename))
            face_image_encoding = face_recognition.face_encodings(face_image)[0]

            index_key = len(self.known_encoding_faces)

            self.known_encoding_faces.append(face_image_encoding)

            index_key_string = str(index_key)
            self.face_user_keys[index_key_string] = user_id

        self.known_encoding_faces2 = self.known_encoding_faces
        self.known_encoding_faces2 = np.asarray(self.known_encoding_faces2)
        self.known_encoding_faces2 = np.reshape(self.known_encoding_faces2, (len(self.known_encoding_faces), 128))
        self.known_encoding_faces2 = self.known_encoding_faces2.astype(np.float32)
        print(self.known_encoding_faces2.shape)

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
            face = {
                "id": row[0],
                "user_id": user_id,
                "filename": filename,
                "created": created
            }
            self.faces.append(face)
            print('name ', name)
            print('filename', filename)
            self.known_face_names.append(name)

            face_image = face_recognition.load_image_file(self.load_train_file_by_name(name, filename))
            face_image_encoding = face_recognition.face_encodings(face_image)[0]

            index_key = len(self.known_encoding_faces)

            self.known_encoding_faces.append(face_image_encoding)

            index_key_string = str(index_key)
            self.face_user_keys[index_key_string] = user_id

        self.known_encoding_faces2 = self.known_encoding_faces
        self.known_encoding_faces2 = np.asarray(self.known_encoding_faces2)
        self.known_encoding_faces2 = np.reshape(self.known_encoding_faces2, (len(self.known_encoding_faces), 128))
        self.known_encoding_faces2 = self.known_encoding_faces2.astype(np.float32)
        


        # print(row)
    def recognize2(self, name, unknown_image_path):
        face_locations = []
        face_encodings = []
        face_names = []
        output = {}
        unknown_image = face_recognition.load_image_file(unknown_image_path)

        # known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)
        unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=1)
        # print(unknown_face_location)ạconda create --name my_env python=3
        # known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=3, model='cnn')
        # unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=3, model='cnn')
        if(len(unknown_face_location) > 1):
            print(unknown_face_location)
            output['code'] = 2
            output['message'] = 'Phát hiện gian lận'
            output['status'] = 'CHEAT'
            return output

        unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location)[0]
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

    def get_path_image_in_db(self, name):
        # print(name)
        path_np = []
        results = self.db.select(
            'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.name = %s',
            [name])

        for i in range(len(results)):
            image_name = results[i][5]
            np_name = image_name+str('.npy')
            path_np.append(path.join(self.storage, 'trained', name, np_name))

        return path_np

    def recognize(self, name, unknown_image_path):
        known_path_np = self.get_path_image_in_db(name)
        face_distance_average = 0
        output = {}
        output['compare'] = []
        for i in range(len(known_path_np)):

            # known_image = face_recognition.load_image_file(known_path_np[i])
            print(known_path_np[i])
            unknown_image = face_recognition.load_image_file(unknown_image_path)

            # known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)
            unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=1)
            # print(unknown_face_location)ạ
            # known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=3, model='cnn')
            # unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=3, model='cnn')
            if(len(unknown_face_location) > 1):
                print(unknown_face_location)
                output['code'] = 2
                output['message'] = 'Phát hiện gian lận'
                output['status'] = 'CHEAT'
                return output
            known_encoding = load(known_path_np[i])
            # known_encoding = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location)[0]
            # known_encoding = face_recognition.face_encodings(known_image, known_face_locations=known_face_location, num_jitters=2)[0]
            # unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location, num_jitters=2)[0]
            compare_faces = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.56)[0]
            face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
            print(face_distance)

            print(type(compare_faces))
            face_distance_average = face_distance_average+face_distance
            print(compare_faces)
            if compare_faces == True:
                output['compare'].append({"message": "Valid", "face_distance": face_distance})

            else:
                output['compare'].append({"message": "Invalid", "face_distance": face_distance})

        # return compare_faces, face_distance
        face_distance_average = face_distance_average/3

        if face_distance_average > 0.56:
            output['message'] = 'Khuôn mặt này không hợp lệ'
            output['face_distance_average'] = face_distance_average
            output['code'] = 2
            output['status'] = 'INVALID'

        else:
            output['message'] = 'Khuôn mặt này  hợp lệ'
            output['face_distance_average'] = face_distance_average
            output['code'] = 1
            output['status'] = 'VALID'

        print(output)
        return output
