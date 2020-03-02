import face_model
import argparse
import cv2
import sys
import numpy as np
import os
from tqdm import *
import imgaug as ia
from imgaug import augmenters as iaa
import facenet


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
    return parser.parse_args(argv)


def sometimes(aug):
    return iaa.Sometimes(0.8, aug)


def main(args):

    # dataset = facenet.get_dataset(args.data_dir)

    # label_person = []
    # # Check that there are at least one training image per class
    # for cls in dataset:
    #     assert(len(cls.image_paths) > 0,
    #            'There must be at least one image for each class in the dataset')
    #     print('class', cls.name)
    #     print('cls', cls.image_paths)
    #     # label_person += len(cls.image_paths)*[cls.name]

    # print('dataset', dataset)

    # paths, labels, label_person = facenet.get_image_paths_and_labels(
    #     dataset)
    # print('paths', paths)
    # print('labels', labels)
    # print('person', label_person)
    # print('Number of classes: %d' % len(dataset))
    # print('Number of images: %d' % len(paths))

    print(args.model)
    model = face_model.FaceModel(args)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        sometimes(
            iaa.OneOf([
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.Add((-20, 20), per_channel=0.5),
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.GaussianBlur((0, 2.0)),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3)),
                iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5))
            ])
        )
    ])
    input_dir = '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4processed/'
    # for mset in ['train', 'test']:
    output_dir = '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4mobile'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for rdir, sdir, files in os.walk(input_dir):
        for file in tqdm(files):
            if '.png' not in file:
                continue
            fn, fe = os.path.splitext(file)
            img_path = os.path.join(rdir, file)
            label_name = rdir.replace(input_dir, '')
            print(label_name)
            # print(file)
            print(img_path)
            # print(fe)

            img_org = cv2.imread(img_path)
            # img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

            img = np.transpose(img_org, (2, 0, 1))
            print('image shape', img.shape)
            emb = model.get_feature(img)
            output_label = output_dir+'/'+label_name
            if not os.path.exists(output_label):
                os.makedirs(output_label)

            np.save(output_label + '/%s.npy' % fn, emb)

            # if mset == 'test':
            #     flip_img = cv2.flip(img_org, 1)
            #     flip_img = np.transpose(flip_img, (2,0,1))
            #     emb = model.get_feature(flip_img)
            #     np.save(output_dir + '/%s_flip.npy'%fn, emb)

            # if 'model-y1-test2' == args.model.split(',')[0].split('/')[-2]:
            #     augmentation_arr = np.array(
            #         [], dtype=np.float32).reshape(0, 128)
            #     for i in range(100):
            #         img_aug = seq.augment_image(img_org)
            #         img_aug = np.transpose(img_aug, (2, 0, 1))
            #         emb = model.get_feature(img_aug)
            #         augmentation_arr = np.vstack(
            #             (augmentation_arr, emb.reshape(1, 128)))
            #     np.save(output_label + '/%s_augmentation.npy' %
            #             fn, augmentation_arr)
            #     print('################################')
            # else:
            #     augmentation_arr = np.array(
            #         [], dtype=np.float32).reshape(0, 512)
            #     for i in range(100):
            #         img_aug = seq.augment_image(img_org)
            #         img_aug = np.transpose(img_aug, (2, 0, 1))
            #         emb = model.get_feature(img_aug)
            #         augmentation_arr = np.vstack(
            #             (augmentation_arr, emb.reshape(1, 512)))
            #     np.save(output_label + '/%s_augmentation.npy' %
            #             fn, augmentation_arr)
            #     print('--------------------------------')


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
