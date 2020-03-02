from numpy.linalg import norm
from numpy import dot
from scipy.spatial import distance
import requests
from sklearn.svm import SVC
import collections
import cv2
import numpy as np
import pickle
import math
import sys
import os
import facenet
import argparse
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf
from keras.layers import Input, concatenate, Conv2D, Activation, BatchNormalization, Dense, Dropout, Flatten, add, Lambda
from keras.models import *
from keras import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam, RMSprop
import keras
import face_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import face_preprocess
from sys import path
from os.path import dirname as dir
# path.append(dir(path[0]))
from retinaface import RetinaFace


# import align.detect_face
# import faiss


# def init_index(known_encoding_faces2):
#     known_encoding_faces2 = known_encoding_faces2.astype(np.float32)
#     d = 128
#     ngpus = faiss.get_num_gpus()

#     print("number of GPUs:", ngpus)

#     cpu_index = faiss.IndexFlatL2(d)

#     gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
#         cpu_index
#     )

#     # add vectors to the index
#     gpu_index.add(known_encoding_faces2)
#     print('index', gpu_index.ntotal)
#     return gpu_index
BATCH_SIZE = 5
EPOCHS = 300
NUMBER_OF_FOLDS = 5
NUMBER_OF_PARTS = 4
INPUT_DIM = 128
NUMBER_OF_CLASSES = 6


def Model():
    model = Sequential()
    model.add(Dense(2048, input_shape=(128,), init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUMBER_OF_CLASSES, init='uniform'))
    model.add(Activation('softmax'))
    return model


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

    URL = "http://192.168.31.131:8000//shot.jpg"

    # Cai dat cac tham so can thiet
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 112
    # CLASSIFIER_PATH = 'Models/facemodel2.pkl'
    VIDEO_PATH = args.path
    # FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load model da train de nhan dien khuon mat - thuc chat la classifier
    # with open(CLASSIFIER_PATH, 'rb') as file:
    #     emb_array_total, label_person = pickle.load(file)
    # print(emb_array_total.shape)
    # print(len(label_person))

    WEIGHTS_BEST = '/home/tmt/Documents/face_recognition/face_recognition_backend/Models/weights/best_weight.hdf5'

    model = Model()
    model.summary()
    model.load_weights(WEIGHTS_BEST)

    gpuid = 0
    detector = RetinaFace(
        '/home/tmt/Documents/insightface/RetinaFace/model/retinaface-R50/', 0, gpuid, 'net3')

    with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/train_data.pkl', 'rb') as file:
        xtrain, ytrain = pickle.load(file)
    # gpu_index = init_index(xtrain)
    # print(gpu_index)
    # print(xtrain.shape)
    # print(ytrain)
    print("Custom Classifier, Successfully loaded")

    # with tf.Graph().as_default():

    #     # Cai dat GPU neu co
    #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #     sess = tf.Session(config=tf.ConfigProto(
    #         gpu_options=gpu_options, log_device_placement=False))

    # with sess.as_default():

    # Load model MTCNN phat hien khuon mat
    print('Loading feature extraction model')
    print(args)
    model_insight = face_model.FaceModel(args)
    with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    # facenet.load_model(FACENET_MODEL_PATH)

    # # Lay tensor input va output
    # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    # embedding_size = embeddings.get_shape()[1]

    # # Cai dat cac mang con
    # pnet, rnet, onet = align.detect_face.create_mtcnn(
    #     sess, "src/align")

    # people_detected = set()
    # person_detected = collections.Counter()

    # Lay hinh anh tu file video
    cap = cv2.VideoCapture(VIDEO_PATH)
    count_frame = 0
    while (cap.isOpened()):
        # Doc tung frame
        ret, frame = cap.read()
        count_frame = count_frame+1
        # img_resp = requests.get(URL)
        # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        # frame = cv2.imdecode(img_arr, -1)

        # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
        thresh = 0.8
        scales = [1024, 1980]
        # img = misc.imread(image_path)
        # img = cv2.imread(frame)
        img = frame
        print('image shape', img.shape)
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

        faces_found = bounding_boxes.shape[0]
        print('face found', faces_found)
        print('bounding boxes', bounding_boxes)
        try:
            # Neu co it nhat 1 khuon mat trong frame
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                #     print('bb1', bb[i][3]-bb[i][1])
                #     print('frame shape', frame.shape[0])
                #     print('bb2', (bb[i][3]-bb[i][1])/frame.shape[0])
                #     # Cat phan khuon mat tim duoc
                #     cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                #     scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                #                         interpolation=cv2.INTER_CUBIC)

                #     # scaled = facenet.prewhiten(scaled)
                #     print('!!!!!!!!!!!!!!!!!')
                #     print(scaled.shape)
                #     print('count frame',count_frame)

                #     # cv2.imwrite(output_image, cropped)

                #     scaled_reshape = scaled.reshape(
                #         3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
                #     print('######################')
                #     print('scale shape', scaled_reshape.shape)

                #     emb = model_insight.get_feature(scaled_reshape)

                nimg = face_preprocess.preprocess(
                    img, bounding_boxes[0, :], landmarks[0, :, :], image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                # output_image='/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/find_face/'+str(count_frame)+'.jpg'
                # print('output_image',output_image)
                # cv2.imwrite(output_image, nimg)
                nimg = np.transpose(nimg, (2, 0, 1))

                emb = model_insight.get_feature(nimg).reshape(1, -1)
                print('emb', emb.shape)
                # print('emb',emb)
                print('------------------')
                print('type', type(emb))

                # emb = emb.reshape(1, 128)

                print('emb shape', emb.shape)
                label_predict = model.predict(emb)
                print(label_predict)
                confidence = np.amax(label_predict)
                print('confidence', np.amax(label_predict))
                # label_predict=np.round(label_predict)
                # print(label_predict)
                inverted = label_encoder.inverse_transform(
                    [np.argmax(label_predict)])
                print('inverted', inverted)
                print('label_predict[0]', label_predict[0])
                name = inverted[0]
                out_simi = '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4mobile/' + \
                    inverted[0]+'/frame 70.npy'
                print(out_simi)
                emb_simi = np.load(out_simi).reshape(128)
                # print('emb_simi',emb_simi)
                print('euclidean: ', np.sum(np.square(emb_simi-emb)))

                cos_sim = dot(emb, emb_simi)/(norm(emb)*norm(emb_simi))
                print('cos_sim', cos_sim)

                # feed_dict = {
                #     images_placeholder: scaled_reshape, phase_train_placeholder: False}
                # emb_array = sess.run(
                #     embeddings, feed_dict=feed_dict)

                # print(emb.shape)
                # emb_array = emb.astype(np.float32)
                # name = 'unknown'
                # k = 2                                       # we want to see 4 nearest neighbors
                # D, I = gpu_index.search(emb_array, k)
                # print('I[0][0]', I[0])

                # best_match_index = I[0][0]

                # print('name', ytrain[best_match_index])
                # d = distance.euclidean(
                #     emb_array, xtrain[best_match_index])
                # print('d',d)
                # print(d)

                # if d <= 0.65:
                #     name = label_person[best_match_index]

                # # Dua vao model de classifier
                # predictions = model.predict_proba(emb_array)
                # best_class_indices = np.argmax(predictions, axis=1)
                # best_class_probabilities = predictions[
                #     np.arange(len(best_class_indices)), best_class_indices]

                # # Lay ra ten va ty le % cua class co ty le cao nhat
                # best_name = class_names[best_class_indices[0]]
                # print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                # Ve khung mau xanh quanh khuon mat
                cv2.rectangle(
                    frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                text_x = bb[i][0]
                text_y = bb[i][3] + 20

                # # Neu ty le nhan dang > 0.5 thi hien thi ten
                # if best_class_probabilities > 0.5:
                #     name = class_names[best_class_indices[0]]
                # else:
                #     # Con neu <=0.5 thi hien thi Unknow
                #     name = "Unknown"

                # Viet text len tren frame
                show_infor=str(name)+'------'+str(confidence)
                color = (0, 0, 255)
                cv2.putText(frame, show_infor, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, color=(0, 0, 255), thickness=1, lineType=2)
                # cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                #             cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #             1, (255, 255, 255), thickness=1, lineType=2)
                # person_detected[best_name] += 1
        except:
            pass

        # Hien thi frame len man hinh
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
