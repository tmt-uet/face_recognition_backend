import pandas as pd
import numpy as np

import keras
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import *
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, Activation, BatchNormalization, Dense, Dropout, Flatten, add, Lambda
import tensorflow as tf
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import random
import threading
from random import randint
import os
from clr_callback import CyclicLR
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

BATCH_SIZE = 5
EPOCHS = 50
NUMBER_OF_FOLDS = 5
NUMBER_OF_PARTS = 4
INPUT_DIM = 128
NUMBER_OF_CLASSES = 6


class ThreadSafeIterator:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(xtrain_fold, ytrain_fold, train_size, batch_size):
    while True:
        xtrain_fold, ytrain_fold = shuffle(xtrain_fold, ytrain_fold)
        for start in range(0, train_size, batch_size):
            end = min(start + batch_size, train_size)
            x_batch = np.array([], dtype=np.float32).reshape(0, 128)
            for i in range(start, end, 1):
                x_batch = np.vstack(
                    (x_batch, xtrain_fold[i, randint(0, 99), :].reshape(1, 128)))
            y_batch = ytrain_fold[start:end, :]
            yield x_batch, y_batch


@threadsafe_generator
def valid_generator(xvalid_fold, yvalid_fold, valid_size, batch_size):
    while True:
        for start in range(0, valid_size, batch_size):
            end = min(start + batch_size, valid_size)
            x_batch = xvalid_fold[start:end, :]
            y_batch = yvalid_fold[start:end, :]
            yield x_batch, y_batch


def get_y_true(df):
    y_true = []
    for index, row in df.iterrows():
        y_true.append(to_categorical(
            row['label'], num_classes=NUMBER_OF_CLASSES))
    return np.array(y_true)


def Model():
    model = Sequential()
    model.add(Dense(2048, input_shape=(128,), init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(NUMBER_OF_CLASSES, init='uniform'))
    model.add(Activation('softmax'))
    return model


if __name__ == '__main__':
    # test_df = pd.read_csv('../../datasets/test_refined.csv')
    # train_df = pd.read_csv('../../datasets/train_refined.csv')

    with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/train_data.pkl', 'rb') as file:
        xtrain, ytrain = pickle.load(file)

    # with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/train_aug_data.pkl', 'rb') as file:
    #     xtrain_aug, ytrain = pickle.load(file)
    if not os.path.exists('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/weights'):
        os.makedirs(
            '/home/tmt/Documents/face_recognition/face_recognition_backend/Models/weights')
    # print(xtrain_aug[0:100, :, :])

    values = np.array(ytrain)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform(
        [np.argmax(onehot_encoded[0, :])])
    print(inverted)
    ytrain = onehot_encoded
    print(ytrain.shape)
    with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/label_encoder.pkl', 'wb') as outfile:
        pickle.dump(label_encoder, outfile)

    # kf = KFold(n_splits=7)  # Define the split - into 7 folds
    # kf.get_n_splits(xtrain)
    # for train_index, test_index in kf.split(xtrain):
    #     print('TRAIN:', train_index, '“TEST:”', test_index)
    #     X_train, X_test = xtrain[train_index], xtrain[test_index]
    #     print('x train', len(X_train))
    #     print('x test', len(X_test))

    #     y_train, y_test = ytrain[train_index], ytrain[test_index]

    # tidxs = 110
    # xtrain_fold = xtrain[0:tidxs, :]
    # ytrain_fold = ytrain[0:tidxs, :]
    # vidxs = 110
    # xvalid_fold = xtrain[vidxs:117, :]
    # yvalid_fold = ytrain[vidxs:117, :]

    # train_size = tidxs
    # valid_size = 117-vidxs

    xtrain_fold, xvalid_fold, ytrain_fold, yvalid_fold = train_test_split(
        xtrain, ytrain, test_size=0.33)
    train_size = len(xtrain_fold)
    valid_size = len(xvalid_fold)

    train_steps = np.ceil(float(train_size) / float(BATCH_SIZE))
    valid_steps = np.ceil(float(valid_size) / float(BATCH_SIZE))
    print('TRAIN SIZE: %d VALID SIZE: %d' % (train_size, valid_size))

    WEIGHTS_BEST = '/home/tmt/Documents/face_recognition/face_recognition_backend/Models/weights/best_weight.hdf5'
    loss_average = 0.0
    acc_average = 0.0
    clr = CyclicLR(base_lr=1e-7, max_lr=1e-3, step_size=6 *
                   train_steps, mode='exp_range', gamma=0.99994)
    early_stopping = EarlyStopping(
        monitor='val_acc', patience=20, verbose=1, mode='max')
    save_checkpoint = ModelCheckpoint(
        WEIGHTS_BEST, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')

    callbacks = [save_checkpoint, early_stopping, clr]

    model = Model()
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    model.fit(xtrain_fold, ytrain_fold, epochs=EPOCHS, validation_data=(xvalid_fold, yvalid_fold),
              callbacks=callbacks)
    model.save_weights(WEIGHTS_BEST)
    score = model.evaluate(x=xvalid_fold, y=yvalid_fold,
                           batch_size=BATCH_SIZE, verbose=1)

    # model.load_weights(WEIGHTS_BEST)
    y_predict = model.predict(xvalid_fold)
    # print(score)
    print(y_predict)
    print(yvalid_fold)

    #     # model.fit_generator(generator=train_generator(xtrain_fold, ytrain_fold, train_size, BATCH_SIZE), steps_per_epoch=train_steps, epochs=EPOCHS, verbose=1,
    #                     validation_data=valid_generator(xvalid_fold, yvalid_fold, valid_size, BATCH_SIZE), validation_steps=valid_steps, callbacks=callbacks)

    # score = model.evaluate(x=xvalid_fold, y=yvalid_fold,
    #                        batch_size=BATCH_SIZE, verbose=1)
    # print(score)
    # loss_average += score[0]
    # acc_average += score[1]
    # print('loss : %f'.format(loss_average))
    # print('accuracy: %f'.format(acc_average))

    # print('loss :', loss_average)
    # xtrain = np.load('train_data.npy')
    # xtrain_aug = np.load('train_aug_data.npy')
    # ytrain = get_y_true(train_df)
    # xtest = np.load('test_data.npy')
    # xtest_flip = np.load('test_flip_data.npy')

    # if not os.path.exists('weights'):
    #     os.makedirs('weights')

    # ptest = np.zeros((xtest.shape[0], NUMBER_OF_CLASSES), dtype=np.float64)
    # training_log = open('training_log.txt', 'w')
    # loss_average = 0.0
    # acc_average = 0.0
    # for part in random.sample(range(30), NUMBER_OF_PARTS):
    #     for fold in range(NUMBER_OF_FOLDS):
    #         v_df = train_df.loc[train_df['rt%d' % part] == fold]
    #         vidxs = v_df.index.values.tolist()
    #         t_df = train_df.loc[~train_df.index.isin(v_df.index)]
    #         tidxs = t_df.index.values.tolist()
    #         print('**************Part %d    Fold %d**************' % (part, fold))

    #         xtrain_fold = xtrain_aug[tidxs, :, :]
    #         ytrain_fold = ytrain[tidxs, :]

    #         xvalid_fold = xtrain[vidxs, :]
    #         yvalid_fold = ytrain[vidxs, :]

    #         train_size = len(tidxs)
    #         valid_size = len(vidxs)
    #         train_steps = np.ceil(float(train_size) / float(BATCH_SIZE))
    #         valid_steps = np.ceil(float(valid_size) / float(BATCH_SIZE))
    #         print('TRAIN SIZE: %d VALID SIZE: %d' % (train_size, valid_size))

    #         WEIGHTS_BEST = 'weights/best_weight_part%d_fold%d.hdf5' % (
    #             part, fold)

    #         clr = CyclicLR(base_lr=1e-7, max_lr=1e-3, step_size=6 *
    #                        train_steps, mode='exp_range', gamma=0.99994)
    #         early_stopping = EarlyStopping(
    #             monitor='val_acc', patience=20, verbose=1, mode='max')
    #         save_checkpoint = ModelCheckpoint(
    #             WEIGHTS_BEST, monitor='val_acc', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
    #         callbacks = [save_checkpoint, early_stopping, clr]

    #         model = Model()
    #         model.summary()
    #         model.compile(loss='categorical_crossentropy',
    #                       optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    #         model.fit_generator(generator=train_generator(xtrain_fold, ytrain_fold, train_size, BATCH_SIZE), steps_per_epoch=train_steps, epochs=EPOCHS, verbose=1,
    #                             validation_data=valid_generator(xvalid_fold, yvalid_fold, valid_size, BATCH_SIZE), validation_steps=valid_steps, callbacks=callbacks)

    #         model.load_weights(WEIGHTS_BEST)

    #         ptest += model.predict(xtest, batch_size=BATCH_SIZE, verbose=1)
    #         ptest += model.predict(xtest_flip,
    #                                batch_size=BATCH_SIZE, verbose=1)

    #         score = model.evaluate(
    #             x=xvalid_fold, y=yvalid_fold, batch_size=BATCH_SIZE, verbose=1)
    #         loss_average += score[0]
    #         acc_average += score[1]
    #         training_log.write('PART:%d FOLD:%d LOSS:%f ACC:%f\n' %
    #                            (part, fold, score[0], score[1]))

    #         K.clear_session()

    # ptest /= float(2*NUMBER_OF_PARTS*NUMBER_OF_FOLDS)
    # np.save('ptest.npy', ptest)

    # loss_average /= float(NUMBER_OF_PARTS*NUMBER_OF_FOLDS)
    # acc_average /= float(NUMBER_OF_PARTS*NUMBER_OF_FOLDS)
    # training_log.write('AVERAGE LOSS:%f ACC:%f\n' %
    #                    (loss_average, acc_average))
    # training_log.close()
