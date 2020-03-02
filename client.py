import cv2
import io
import socket
import struct
import time
import pickle
import zlib
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('0.0.0.0', 8000))
    
    connection = client_socket.makefile('wb')
    
    cam = cv2.VideoCapture(0)
    print('--------------')
    cam.set(3, 320)
    cam.set(4, 240)

    img_counter = 0

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    print('--------------')
    while True:
        ret, frame = cam.read()
        result, frame = cv2.imencode('.jpg', frame, encode_param)
    #    data = zlib.compress(pickle.dumps(frame, 0))
        data = pickle.dumps(frame, 0)
        size = len(data)

        print("{}: {}".format(img_counter, size))
        client_socket.sendall(struct.pack(">L", size) + data)
        img_counter += 1

    cam.release()
except Exception as e:
    client_socket.close()
    print(e)
    print("finished get frame")
