import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
from mtcnn.mtcnn import MTCNN
import face_preprocess
thresh = 0.8
scales = [1024, 1980]

# count = 1

gpuid = 0
detector = RetinaFace(
    '/home/tmt/Documents/insightface/RetinaFace/model/retinaface-R50/', 0, gpuid, 'net3')

img = cv2.imread(
    '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4/Ha/frame 50.jpg')
print(img.shape)
im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
#im_scale = 1.0
# if im_size_min>target_size or im_size_max>max_size:
im_scale = float(target_size) / float(im_size_min)
# prevent bigger axis from being more than max_size:
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

print('im_scale', im_scale)

scales = [im_scale]
flip = False
img = img[:, :, 0:3]
bounding_boxes, landmarks = detector.detect(
    img, thresh, scales=scales, do_flip=flip)
nrof_faces = bounding_boxes.shape[0]
print(bounding_boxes)
print(landmarks)
print(landmarks[0,:,:])
nimg = face_preprocess.preprocess(
img, bounding_boxes[0,:], landmarks[0,:,:], image_size='112,112')
output_filename_n='/home/tmt/Documents/insightface/RetinaFace/detector_test3.jpg'
# misc.imsave(output_filename_n, nimg)
cv2.imwrite(output_filename_n, nimg)


# if nrof_faces == 1:
#     nimg = face_preprocess.preprocess(
#     img, bounding_boxes[0,:], landmarks.reshape(5,2), image_size='112,112')
#     nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
#     output_filename_n='/home/tmt/Documents/insightface/RetinaFace/detector_test3.jpg'
#     # misc.imsave(output_filename_n, nimg)
#     cv2.imwrite(output_filename_n, nimg)


# for c in range(count):
#     faces, landmarks = detector.detect(
#         img, thresh, scales=scales, do_flip=flip)
#     print(c, faces.shape, landmarks.shape)

# detector_mtcnn = MTCNN()
# bboxes = detector_mtcnn.detect_faces(img)
# faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
# print('facess', faces)
# print('landmark', landmarks)

# # test MTCNN

# if len(bboxes) != 0:
#     for bboxe in bboxes:
#         bbox = bboxe['box']
#         bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
#         landmarks = bboxe['keypoints']
#         # print(landmarks)
#         landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
#                               landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
#         landmarks = landmarks.reshape((2, 5)).T
#         print(landmarks)

# if faces is not None:
#     print('find', faces.shape[0], 'faces')
#     for i in range(faces.shape[0]):
#         #print('score', faces[i][4])
#         box = faces[i].astype(np.int)
#         #color = (255,0,0)
#         color = (0, 0, 255)
#         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
#         print('box', box)
#         if landmarks is not None:
#             landmark5 = landmarks[i].astype(np.int)
#             # print(landmark.shape)
#             for l in range(landmark5.shape[0]):
#                 color = (0, 0, 255)
#                 if l == 0 or l == 3:
#                     color = (0, 255, 0)
#                 cv2.circle(
#                     img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

# filename = 'detector_test.jpg'
# print('writing', filename)
# cv2.imwrite(filename, img)


# scales = [1024, 1980]

# img = cv2.imread(
#     '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4/Hoang/frame 10.jpg')
# print(img.shape)
# im_shape = img.shape
# target_size = scales[0]
# max_size = scales[1]
# im_size_min = np.min(im_shape[0:2])
# im_size_max = np.max(im_shape[0:2])
# #im_scale = 1.0
# # if im_size_min>target_size or im_size_max>max_size:
# im_scale = float(target_size) / float(im_size_min)
# # prevent bigger axis from being more than max_size:
# if np.round(im_scale * im_size_max) > max_size:
#     im_scale = float(max_size) / float(im_size_max)

# print('im_scale', im_scale)

# scales = [im_scale]
# flip = False

# # for c in range(count):
# #     faces, landmarks = detector.detect(
# #         img, thresh, scales=scales, do_flip=flip)
# #     print(c, faces.shape, landmarks.shape)

# faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
# print('facess', faces)
