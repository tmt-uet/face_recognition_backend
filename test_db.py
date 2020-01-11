import face_recognition
from os import path
import numpy as np
import requests
from my_sql_db import Database
import shutil

from mtcnn import MTCNN
import cv2
import time
# db = Database()
# db.init_db()
# name = 'abc'
# # results = db.select(
# #     'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.name = %s',
# #     [name])
# # re = db.select('SELECT id from users WHERE name= %s', [name])[0][0]
# # # print(results[0][5])
# # print(results)
# check_exist = db.select('SELECT count(*) from users WHERE name=%s', [name])
# # print(check_exist)
# check = db.select('SELECT * from users WHERE name=%s', [name])
# if (len(check)) > 0:
#     print("YES")
# print('len check', len(check))

# print("check exist", check_exist)

shutil.rmtree('/home/tmt/Documents/face/collection/Agent 1')


created1 = int(time.time())

img = cv2.cvtColor(cv2.imread(
    "/home/tmt/Documents/face/collection/Cong Anh/53491942_1224175431066305_4033200567001022464_n.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
created2 = int(time.time())
print(created2-created1)

result = detector.detect_faces(img)
bounding_box = result[0]['box']
# print('boundinggggggggggggggggggggggg', result)
# crop_img = img[bounding_box[0]:bounding_box[0]+bounding_box[2], bounding_box[1]:bounding_box[1]+bounding_box[3]]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)


def convert_to_list_tuple(bounding_box):
    list_tuple = []
    convert_tuple = (bounding_box[0], bounding_box[1]+bounding_box[3], bounding_box[0]+bounding_box[2], bounding_box[1])
    list_tuple.append(convert_tuple)
    return list_tuple


list_tuple = convert_to_list_tuple(bounding_box)
print("list tuple", list_tuple)


known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/Cong Anh/53491942_1224175431066305_4033200567001022464_n.jpg')

created1 = int(time.time())
known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)
print("known_face_location", known_face_location)
known_encoding = face_recognition.face_encodings(known_image, known_face_locations=list_tuple)[0]


# img = cv2.cvtColor(cv2.imread(
#     "/home/tmt/Documents/face/collection/Cong Anh/53491942_1224175431066305_4033200567001022464_n.jpg"), cv2.COLOR_BGR2RGB)
# crop_img = img[known_face_location[0][0]:known_face_location[0][2], known_face_location[0][3]:known_face_location[0][1]]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
