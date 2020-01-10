import face_recognition
from os import path
import numpy as np
import requests
from flask import Flask, json


class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.db = app.db

    def get_path_image_in_db(self, name):
        # print(name)
        path_image = []
        results = self.db.select(
            'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.name = %s',
            [name])

        for i in range(len(results)):
            image_name = results[i][5]
            path_image.append(path.join(self.storage, 'trained', name, image_name))

        return path_image

    def recognize(self, name, unknown_image_path):
        known_path_image = self.get_path_image_in_db(name)
        face_distance_average = 0
        output = {}
        output['compare'] = []
        for i in range(len(known_path_image)):

            known_image = face_recognition.load_image_file(known_path_image[i])
            print(known_path_image[i])
            unknown_image = face_recognition.load_image_file(unknown_image_path)

            known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)
            unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=1)
            # known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=3, model='cnn')
            # unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=3, model='cnn')
            if(len(unknown_face_location) > 1):
                output['code'] = 2
                output['message'] = 'Phát hiện gian lận'
                output['status'] = 'CHEAT'
                return output
            known_encoding = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location)[0]
            # known_encoding = face_recognition.face_encodings(known_image, known_face_locations=known_face_location, num_jitters=2)[0]
            # unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location, num_jitters=2)[0]
            compare_faces = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.56)[0]
            face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]

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
