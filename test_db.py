import face_recognition
from os import path
import numpy as np
import requests
from my_sql_db import Database
import shutil
from PIL import Image
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

# shutil.rmtree('/home/tmt/Documents/face/collection/Agent 1')


created1 = int(time.time())

img = cv2.cvtColor(cv2.imread(
    "/home/tmt/Documents/face/collection/TMT/beauty_20191009001401.jpg"), cv2.COLOR_BGR2RGB)
print(img.shape)


def resized(img):
    print("Original Dimensions", img.shape)
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print("Resized Dimensions", img.shape)
    return resized


def detect(img):

    detector = MTCNN()
    created2 = int(time.time())
    print(created2-created1)

    result = detector.detect_faces(img)
    bounding_box = result[0]['box']
    print('boundinggggggggggggggggggggggg', result)
    # crop_img = img[169:795, 41:536]
    crop_img = img[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    print("Crop image size", crop_img.shape)
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)


def painting():
    detector = MTCNN()

    image = cv2.cvtColor(cv2.imread("/home/tmt/Documents/face/collection/TMT/beauty_20191009001401.jpg"), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    print('boundinggggggggggggggggggggggg', result)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)

    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    # cv2.imwrite("ivan_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow("cropped", image)
    cv2.waitKey(0)


# painting()

def face_recog():
        # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("/home/tmt/Documents/face/collection/TMT/beauty_20191009001401.jpg")

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    face_locations = face_recognition.face_locations(image)
    print(face_locations[0])
    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        # crop_img = img[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]

        pil_image = Image.fromarray(face_image)
        pil_image.show()


face_recog()
# detect(img)
painting()
x = 18
y = 111
width = 56
height = 142

# (top=y=111, right=x+width=18+56, bottom=y+height=111+142, left=x=18)

(220, 165, 3)

[(80, 127, 187, 20)]

# top = bounding_box[1]
# bottom = bounding_box[1]+bounding_box[3]
# left = bounding_box[0]
# right = bounding_box[0]+bounding_box[2]



# [x, y, width, height]
# [1, 93, 570, 752]
# left=x,right=x+width,top=y,bot=y+height




# (top, right, bottom, left)
# [(233, 603, 788, 48)]

# def convert_to_list_tuple(bounding_box):
#     list_tuple = []
#     convert_tuple = (bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3], bounding_box[0])
#     list_tuple.append(convert_tuple)
#     return list_tuple


# list_tuple = convert_to_list_tuple(bounding_box)
# print("list tuple", list_tuple)


# known_image = face_recognition.load_image_file('/home/tmt/Documents/face/collection/Cong Anh/53491942_1224175431066305_4033200567001022464_n.jpg')

# created1 = int(time.time())
# known_face_location = face_recognition.face_locations(known_image, number_of_times_to_upsample=1)
# print("known_face_location", known_face_location)
# known_encoding = face_recognition.face_encodings(known_image, known_face_locations=list_tuple)[0]

# known_face_location = [(194, 707, 657, 245), (563, 218, 718, 64)]

# img = cv2.cvtColor(cv2.imread(
#     "/home/tmt/Documents/face/collection/Nancy/nancy-momoland-guong-mat-sang-gia-cua-kpop.jpg"), cv2.COLOR_BGR2RGB)
# crop_img = img[known_face_location[0][0]:known_face_location[0][2], known_face_location[0][3]:known_face_location[0][1]]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
