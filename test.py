import face_recognition
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import pickle
# img = Image.open('/home/tmt/Documents/face_recognition/my_app/storage/trained/Tung/1578372281beauty_20191009001401.jpg').convert('LA')
# img.save('/home/tmt/Documents/face_recognition/my_app/storage/trained/1578372281beauty_20191009001401.png')

import cv2

# originalImage = cv2.imread('/home/tmt/Documents/face_recognition/my_app/storage/trained/Tung/1578372281beauty_20191009001401.jpg')
# grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

# (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

# cv2.imshow('Black white image', blackAndWhiteImage)
# cv2.imshow('Original image', originalImage)
# cv2.imshow('Gray image', grayImage)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('/home/tmt/Documents/face_recognition/my_app/storage/trained/1578372281beauty_20191009001401.jpg', grayImage)


# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# img = mpimg.imread('lena.png')

# gray = rgb2gray(img)

# plt.imshow(gray, cmap = plt.get_cmap('gray'))

# plt.savefig('lena_greyscale.png')
# plt.show()

# known_image = face_recognition.load_image_file('/home/tmt/Documents/face_recognition/my_app/storage/trained/1578372281beauty_20191009001401.jpg')

# created1 = int(time.time())
# known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1, model='cnn')

# print(known_face_location)
# known_encoding = face_recognition.face_encodings(known_image, known_face_locations=known_face_location)[0]
# print(type(known_encoding))
# print(known_encoding.shape)
# created2 = int(time.time())
# print(created2-created1)

# known_image2 = face_recognition.load_image_file('/home/tmt/Documents/face_recognition/my_app/storage/trained/Tung/1578372281beauty_20191009001401.jpg')

# created3 = int(time.time())
# known_face_location2 = face_recognition.face_locations(known_image2, number_of_times_to_upsample=1, model='cnn')
# known_encoding2 = face_recognition.face_encodings(known_image, known_face_locations=known_face_location2)[0]
# created4 = int(time.time())
# print(created4-created3)



with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/facemodel.pkl', 'rb') as infile:
    (emb_array, label_person) = pickle.load(infile)
print(label_person)
