import face_recognition
from numpy import save
from numpy import load
import os
known_image = face_recognition.load_image_file("/home/tmt/Documents/face_recognition/my_app/storage/beauty_20191009001401_test.jpg")
biden_encoding = face_recognition.face_encodings(known_image)[0]
print(biden_encoding)
# save('/home/tmt/Documents/face_recognition/my_app/storage/trained/data.npy', biden_encoding)


# data = load('/home/tmt/Documents/face_recognition/my_app/storage/trained/data.npy')
# print(data)
# os.remove('/home/tmt/Documents/face_recognition/my_app/storage/trained/beauty_20190915092027.jpg')
# /home/tmt/Documents/face_recognition/my_app/storage/beauty_20190915092027.jpg
