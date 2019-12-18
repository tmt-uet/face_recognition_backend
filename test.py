import urllib.request
import cv2
import numpy as np
import time
import requests
import face_recognition

face_image = face_recognition.load_image_file('/home/tmt/Documents/face_recognition/my_app/storage/trained/43358318_1119791154838067_3691811964742270976_n.jpg')
try:
    face_image_encoding = face_recognition.face_encodings(face_image)[0]
except:
    print("not found face in image")
# print(type(face_image_encoding))
