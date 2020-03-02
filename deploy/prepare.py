import pandas as pd
import numpy as np
import os
from tqdm import *
from multiprocessing import Pool, cpu_count
import pickle


class Pair:
    def __init__(self, image_path, label_name):
        self.image_path = image_path
        self.label_name = label_name

# def my_process1(file_name):
#     emb_path = '../../models/insightface/embedding/model-r100-ii/test/%s' % file_name.replace(
#         '.png', '.npy')
#     emb = np.load(emb_path).reshape(512)
#     return emb


# def my_process2(file_name):
#     emb_path = '../../models/insightface/embedding/model-r100-ii/test/%s' % file_name.replace(
#         '.png', '_flip.npy')
#     emb = np.load(emb_path).reshape(512)
#     return emb


def my_process3(file_name):
    # emb_path = '../../models/insightface/embedding/model-r100-ii/train/%s' % file_name.replace(
    #     '.png', '.npy')
    emb = np.load(file_name).reshape(128)
    return emb


def my_process4(file_name):

    # emb_path = '../../models/insightface/embedding/model-r100-ii/train/%s' % file_name.replace(
    #     '.png', '_augmentation.npy')
    emb = np.load(file_name).reshape(100, 128)
    return emb


if __name__ == '__main__':
    input_dir = '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4mobile/'
    # for mset in ['train', 'test']:
    # output_dir = '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4mobile'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    list_path = []
    list_path_augmentation = []
    list_name = []
    list_name_augmentation = []
    for rdir, sdir, files in os.walk(input_dir):
        for file in tqdm(files):
            if '_augmentation.npy' in file:
                # np_path = os.path.join(rdir, file)
                # label_name = rdir.replace(input_dir, '')
                # list_path_augmentation.append(np_path)
                # list_name_augmentation.append(label_name)
                # print(label_name)
                # # print(file)
                # print(np_path)
                continue

            else:
                # fn, fe = os.path.splitext(file)
                np_path = os.path.join(rdir, file)
                label_name = rdir.replace(input_dir, '')
                list_path.append(np_path)
                list_name.append(label_name)
                print(label_name)
                # print(file)
                print(np_path)

    dic = {'list_path': list_path, 'list_name': list_name}
    df = pd.DataFrame(dic)
    for element in set(list_name):
        print(element)


    print(df)
    print('list name', len(list_name))
    print('list name augmentation', len(list_name_augmentation))
    print('list path', len(list_path))
    print('list path augmentation', len(list_path_augmentation))

    p = Pool(16)
    train_data = p.map(
        func=my_process3, iterable=list_path)
    p.close()
    test1 = my_process3(list_path[80])
    test2 = train_data[80]
    train_data = np.array(train_data)

    print(train_data.shape)
    # np.save('train_data.npy', train_data)
    with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/train_data.pkl', 'wb') as outfile:
        pickle.dump((train_data, list_name), outfile)

    # print(train_data)
    # train_data = []

    # p = Pool(16)
    # train_aug_data = p.map(
    #     func=my_process4, iterable=list_path_augmentation)
    # p.close()
    # train_aug_data = np.array(train_aug_data)
    # print(train_aug_data.shape)
    # # np.save('train_aug_data.npy', train_aug_data)
    # with open('/home/tmt/Documents/face_recognition/face_recognition_backend/Models/train_aug_data.pkl', 'wb') as outfile:
    #     pickle.dump((train_aug_data, list_name_augmentation), outfile)

    # print(train_aug_data)
    # train_aug_data = []

    # print(test1)
    # print(test2)
    # test_df = pd.read_csv('../../datasets/test_refined.csv')
    # train_df = pd.read_csv('../../datasets/train_refined.csv')

    # p = Pool(16)
    # test_data = p.map(func=my_process1, iterable=test_df.image.values.tolist())
    # p.close()
    # test_data = np.array(test_data)
    # print(test_data.shape)
    # np.save('test_data.npy', test_data)
    # test_data = []

    # p = Pool(16)
    # test_flip_data = p.map(
    #     func=my_process2, iterable=test_df.image.values.tolist())
    # p.close()
    # test_flip_data = np.array(test_flip_data)
    # print(test_flip_data.shape)
    # np.save('test_flip_data.npy', test_flip_data)
    # test_flip_data = []

    # p = Pool(16)
    # train_data = p.map(
    #     func=my_process3, iterable=train_df.image.values.tolist())
    # p.close()
    # train_data = np.array(train_data)
    # print(train_data.shape)
    # np.save('train_data.npy', train_data)
    # train_data = []

    # p = Pool(16)
    # train_aug_data = p.map(
    #     func=my_process4, iterable=train_df.image.values.tolist())
    # p.close()
    # train_aug_data = np.array(train_aug_data)
    # print(train_aug_data.shape)
    # np.save('train_aug_data.npy', train_aug_data)
    # train_aug_data = []


# from multiprocessing import Pool

# import time

# work = (["A", 5], ["B", 2], ["C", 1], ["D", 3])


# def work_log(work_data):
#     print(" Process %s waiting %s seconds" % (work_data[0], work_data[1]))
#     time.sleep(int(work_data[1]))
#     print(" Process %s Finished." % work_data[0])
#     return work_data


# def pool_handler():
#     p = Pool(2)
#     a = p.map(work_log, work)
#     print('a', a)


# if __name__ == '__main__':
#     pool_handler()
