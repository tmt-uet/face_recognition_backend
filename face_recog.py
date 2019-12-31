import face_recognition
from os import path
import numpy as np
import requests


class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.db = app.db

    def get_path_image_in_db(self, name):
        # print(name)

        results = self.db.select(
            'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.name = %s',
            [name])
        image_name = results[0][5]
        path_image = path.join(self.storage, 'trained', image_name)

        return path_image

    def recognize(self, name, unknown_image_path):
        known_path_image = self.get_path_image_in_db(name)

        known_image = face_recognition.load_image_file(known_path_image)
        unknown_image = face_recognition.load_image_file(unknown_image_path)

        known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)
        unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=1)

        known_encoding = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location)[0]

        compare_faces = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.56)[0]
        face_distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        print(type(compare_faces))

        print(compare_faces)
        return compare_faces, face_distance
