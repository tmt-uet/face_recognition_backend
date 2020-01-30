import face_recognition
from os import path
import numpy as np
import requests
import cv2
import faiss


class Live_Face:
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
        self.init_index()

        # self.live_recognize()

    def init_index(self):
        d = 128
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)

        cpu_index = faiss.IndexFlatL2(d)

        self.gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
        )

        self.gpu_index.add(self.known_encoding_faces2)              # add vectors to the index
        print('index', self.gpu_index.ntotal)

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

        # print(row)

    def recognize(self, unknown_filename):
        unknown_image = face_recognition.load_image_file(self.load_unknown_file_by_name(unknown_filename))
        unknown_image_encoding = face_recognition.face_encodings(unknown_image)[0]

        results = face_recognition.compare_faces(self.known_encoding_faces, unknown_image_encoding)
        print("results", results)

        index_key = 0
        for matched in results:

            if matched:
                # so we found this user with index key and find him
                user_id = self.load_user_by_index_key(index_key)

                return user_id

            index_key = index_key + 1

        return None

    def live_recognize(self):

        # video_capture = cv2.VideoCapture(0)
        URL = "http://192.168.1.13:8000//shot.jpg"

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        # known_face_names = [
        #     "Trang",
        #     "Tung",
        #     "Trung",
        #     "Sinh"
        # ]
        while True:
            # Grab a single frame of video
            img_resp = requests.get(URL)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_arr, -1)
            # cv2.imshow("android cam", img)

            # ret, frame = video_capture.read()

            # cv2.imshow("android cam", img)
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)

                    # matches = face_recognition.compare_faces(self.known_encoding_faces, face_encoding)
                    known_encoding_faiss = np.reshape(face_encoding, (1, 128))
                    known_encoding_faiss = known_encoding_faiss.astype(np.float32)

                    name = "Unknown"
                    k = 2                                       # we want to see 4 nearest neighbors
                    D, I = self.gpu_index.search(known_encoding_faiss, k)
                    print('I[0][0]', I[0][0])
                    # print(I)
                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = self.n face with the smallest distance to the new face
                    # face_distances = face_recognition.face_distance(self.known_encoding_faces, face_encoding)
                    # best_match_index = np.argmin(face_distances)
                    # print('best_match_index', best_match_index)

                    # if matches[best_match_index]:
                    #     name = self.known_face_names[best_match_index]

                    best_match_index = I[0][0]
                    name = self.known_face_names[best_match_index]
                    print('name', name)
                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        # video_capture.release()
        cv2.destroyAllWindows()
