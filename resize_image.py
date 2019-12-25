import cv2
import face_recognition
image = cv2.imread('/home/tmt/Documents/face_recognition/my_app/storage/trained/beauty_20191009001401.jpg')
small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
rgb_small_frame = small_frame[:, :, ::-1]
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)[0]
print(face_encodings)
# cv2.imshow('Video', rgb_small_frame)

cv2.imwrite('/home/tmt/Documents/face_recognition/my_app/storage/beauty_20191009001401_test.jpg', rgb_small_frame)
