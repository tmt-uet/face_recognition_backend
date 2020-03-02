import cv2
import sys
import numpy as np
import datetime
import os
import glob
from mtcnn.mtcnn import MTCNN
import argparse
import face_preprocess
import face_model
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))
from RetinaFace.retinaface import RetinaFace

def parse_arguments(argv):

    parser = argparse.ArgumentParser(description='face model test')
    # general
    # parser.add_argument(
    #     '--data_dir', default='/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4processed', help='')
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
    parser.add_argument(
        '--path', help='Path of the video you want to test on.', default=0)
    return parser.parse_args(argv)


def main(args):
    model_insight = face_model.FaceModel(args)
    thresh = 0.8
    scales = [1024, 1980]

    # count = 1

    gpuid = 0
    detector = RetinaFace(
        '/home/tmt/Documents/insightface/RetinaFace/model/retinaface-R50/', 0, gpuid, 'net3')

    img = cv2.imread(
        '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4/Hoang/frame 100.jpg')
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

    # for c in range(count):
    #     faces, landmarks = detector.detect(
    #         img, thresh, scales=scales, do_flip=flip)
    #     print(c, faces.shape, landmarks.shape)
    detector_mtcnn = MTCNN()
    bboxes = detector_mtcnn.detect_faces(img)
    print('bboxes',bboxes)
    faces, landmarks = detector.detect(
        img, thresh, scales=scales, do_flip=flip)
    print('facess', faces)
    print('type face',faces.shape)
    print('landmark', landmarks)
    print('type landmark',landmarks.shape)
    # test MTCNN
    print(faces[0,:])
    print(landmarks.reshape(5,2))
    print('------------------------------')
    nimg = face_preprocess.preprocess(
        img, faces[0,:], landmarks.reshape(5,2), image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    # output_image='/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/find_face/test.jpg'
    # print('output_image',output_image)
    # cv2.imwrite(output_image, nimg)

    nimg = np.transpose(nimg, (2, 0, 1))
    embedding1 = model_insight.get_feature(nimg).reshape(1, -1)
    print('shape',embedding1.shape)


    if len(bboxes) != 0:
        for bboxe in bboxes:
            bbox = bboxe['box']
            bbox = np.array(
                [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
            landmarks = bboxe['keypoints']
            # print(landmarks)
            landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                  landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
            landmarks = landmarks.reshape((2, 5)).T
            print(landmarks)
            print(bbox)
            nimg = face_preprocess.preprocess(
                img, bbox, landmarks, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            embedding2 = model_insight.get_feature(nimg).reshape(1, -1)
            print(embedding2)

    dist = np.sum(np.square(embedding1-embedding2))
    print(dist)
    sim = np.dot(embedding1, embedding2.T)
    print(sim)

    # nimg = face_preprocess.preprocess(
    # img, faces[0,:], landmarks[0,:,:], image_size='112,112')
    # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    # nimg = np.transpose(nimg, (2, 0, 1))
    # embedding = model_insight.get_feature(nimg).reshape(1, -1)
    # print(embedding)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
