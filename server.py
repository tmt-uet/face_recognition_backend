import socket
import sys
import cv2
import pickle
import numpy as np
import struct  # new
import zlib
import argparse
from os import path
import os


def main(args):

    # path_image = '/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image'
    # os.mkdir(path.join('/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image', args.class_user))
    # os.mkdir(path.join('/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image/4', args.user))

    path_image = path.join('/home/tmt/Documents/face_recognition/face_recognition_backend/capture_image', args.class_user, args.user)
    print(path_image)
    HOST = '0.0.0.0'
    PORT = 8000

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')

    s.bind((HOST, PORT))
    print('Socket bind complete')
    s.listen(10)
    print('Socket now listening')

    conn, addr = s.accept()

    img_counter = 0
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    while True:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += conn.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        img_counter += 1
        if img_counter >= 50:
            print('start get image')
            if img_counter % 5 == 0:
                cv2.imwrite(path_image+"/frame %d.jpg" % img_counter, frame)
                print("write frame %d" % img_counter)
        # If image taken reach 100, stop taking video
        if img_counter >= 200:
            print("Successfully Captured")
            break
        cv2.imshow('ImageWindow', frame)
        cv2.waitKey(1)

    s.close()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('class_user', type=str,
                        help='class name', default='3')

    parser.add_argument('user', type=str,
                        help='user name', default='abc')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
