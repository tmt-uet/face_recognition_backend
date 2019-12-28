import face_recognition
from numpy import save
from numpy import load
import os
from my_sql_db import Database
# known_image = face_recognition.load_image_file("/home/tmt/Documents/face_recognition/my_app/storage/beauty_20191009001401_test.jpg")
# biden_encoding = face_recognition.face_encodings(known_image)[0]
# print(biden_encoding)
# save('/home/tmt/Documents/face_recognition/my_app/storage/trained/data.npy', biden_encoding)


# data = load('/home/tmt/Documents/face_recognition/my_app/storage/trained/data.npy')
# print(data)
# os.remove('/home/tmt/Documents/face_recognition/my_app/storage/trained/beauty_20190915092027.jpg')
# /home/tmt/Documents/face_recognition/my_app/storage/beauty_20190915092027.jpg
# db = Database()
# result = db.select('SELECT id,name FROM users WHERE name = %s', ['Trang'])
# print(type(result[0][1]))
# print(result[0][0])


import face_recognition
import time
start_time = time.time()

known_image = face_recognition.load_image_file("/home/tmt/Documents/face_recognition/my_app/storage")
# unknown_image = face_recognition.load_image_file("/home/tmt/Documents/face/collection/TMT/IMG_20190912_172925.jpg")
unknown_image = face_recognition.load_image_file("/home/tmt/Documents/face/collection/Cong Anh/43358318_1119791154838067_3691811964742270976_n.jpg")

print('location')
tung_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)
unknown_face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=1)

print('encoding')
# tung_encoding = face_recognition.face_encodings(known_image, known_face_locations=tung_face_location)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location)[0]

tung_encoding = face_recognition.face_encodings(known_image, known_face_locations=tung_face_location)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=unknown_face_location)[0]

# results = face_recognition.face_distance([tung_encoding], unknown_encoding)
# print(results)
# print(face_recognition.compare_faces([tung_encoding], unknown_encoding, tolerance=0.56))
# print("--- %s seconds ---" % (time.time() - start_time))
