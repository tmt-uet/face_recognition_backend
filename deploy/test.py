import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument(
    '--model', default='/home/tmt/Documents/insightface/models/model-y1-test2/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int,
                    help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int,
                    help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24,
                    type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread(
    '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4/Hoang/frame 70.jpg')
img = model.get_input(img)              # using mtcnn to alignment
f1 = model.get_feature(img)
# print(f1[0:10])
print(f1)
img2 = cv2.imread(
    '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4processed/Hoang/frame 70.png')
# img2 = model.get_input(img2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = np.transpose(img2, (2, 0, 1))
f2 = model.get_feature(img2)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
