from flask import Flask, json, Response, request, render_template, send_file
from werkzeug.utils import secure_filename
from os import path, getcwd
import os
import time
from my_sql_db import Database
from face_recog import Face
import face_recognition
import mysql.connector
from face import Live_Face
import urllib
from urllib.request import urlopen
import io
import requests
import shutil
from flask_cors import CORS
from numpy import save

from mtcnn import MTCNN
import cv2

app = Flask(__name__)


def init_app():
    CORS(app)
    cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

    if os.path.exists(path.join(getcwd(), 'storage')) == False:
        os.mkdir(path.join(getcwd(), 'storage'))

    app.config['storage'] = path.join(getcwd(), 'storage')

    if os.path.exists(path.join(app.config['storage'], 'trained')) == False:
        os.mkdir(path.join(app.config['storage'], 'trained'))

    if os.path.exists(path.join(app.config['storage'], 'unknown')) == False:
        os.mkdir(path.join(app.config['storage'], 'unknown'))

    if os.path.exists(path.join(app.config['storage'], 'model')) == False:
        os.mkdir(path.join(app.config['storage'], 'model'))

    if os.path.exists(path.join(app.config['storage'], 'check_user')) == False:
        os.mkdir(path.join(app.config['storage'], 'check_user'))

    app.config['file_allowed'] = ['image/png', 'image/jpeg']
    app.config['trained'] = path.join(getcwd(), 'storage', 'trained')
    app.config['unknown'] = path.join(getcwd(), 'storage', 'unknown')
    app.config['model'] = path.join(getcwd(), 'storage', 'model')
    app.db = Database()
    app.face = Face(app)

    print('init app done')


init_app()


# app.live_face = Live_Face(app)


def success_handle(code, error_message,  status, mimetype='application/json'):
    return Response(json.dumps({"code": code, "message": error_message, "status": status}), mimetype=mimetype)


def error_handle(code, error_message,  status, mimetype='application/json'):
    return Response(json.dumps({"code": code, "message": error_message, "status": status}),  mimetype=mimetype)


def get_user_by_name(name):
    print("get_user_by_name")
    # user_id = app.db.select('SELECT id from users WHERE name= %s', [name])[0][0]
    user = {}
    results = app.db.select(
        'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.name = %s',
        [name])

    index = 0
    for row in results:
        # print(row)
        face = {
            "id": row[3],
            "user_id": row[4],
            "filename": row[5],
            "created": row[6],
        }
        if index == 0:
            user = {
                "id": row[0],
                "name": row[1],
                "created": row[2],
                "faces": [],
            }
        if row[3]:
            user["faces"].append(face)
        index = index + 1

    if 'id' in user:
        return user
    return None


def remove_path_image(name):
    # shutil.rmtree(path.join(app.config['trained'], name))
    folder_name = path.join(app.config['trained'], name)
    file_list = [f for f in os.listdir(folder_name)]
    for f in file_list:
        os.remove(os.path.join(folder_name, f))

    if os.path.exists(path.join(app.config['unknown'], name)) == True:
        # shutil.rmtree(path.join(app.config['unknown'], name))
        folder_name_unknown = path.join(app.config['unknown'], name)
        file_list = [f for f in os.listdir(folder_name_unknown)]
        for f in file_list:
            os.remove(os.path.join(folder_name_unknown, f))


def delete_user_by_name(name):
    user_id = app.db.select('SELECT id from users WHERE name= %s', [name])[0][0]
    print(user_id)
    app.db.delete('DELETE FROM faces WHERE faces.user_id = %s', [user_id])

    app.db.delete('DELETE FROM users WHERE users.id = %s', [user_id])


def check_request_containt_image_file(request_file):
    if 'file' not in request_file:
        return 1
    file = request_file['file']
    if file.mimetype not in app.config['file_allowed']:
        return 2


def check_url_is_image(url):

    f_ext = os.path.splitext(url)[-1]
    print(f_ext)

    if str(f_ext) != '.jpg' and str(f_ext) != '.png':
        return 1


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api', methods=['GET'])
def homepage():
    return success_handle(1, "OK", "OK")

# @app.route('/api/identify',methos=['POST'])
# def identify():


def check_user_is_exist(name):
    check_exist = app.db.select('SELECT * from users WHERE name=%s', [name])
    print(check_exist)

    if (len(check_exist)) >= 1:
        print("User is exist")
        return 1
    return 0


def checkIP(ip):
    IP = ['125.235.4.59', '127.0.0.1']
    for ip_index in IP:
        if ip == ip_index:
            return 1
    return 0


def extract_face_from_image(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    result = app.detector.detect_faces(img)
    bounding_box = result[0]['box']
    list_tuple = []
    # convert_tuple = (bounding_box[0], bounding_box[1]+bounding_box[3], bounding_box[0]+bounding_box[2], bounding_box[1])
    convert_tuple = (bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3], bounding_box[0])
    list_tuple.append(convert_tuple)
    return list_tuple
    # print('boundinggggggggggggggggggggggg', result)
    # crop_img = img[bounding_box[0]:bounding_box[0]+bounding_box[2], bounding_box[1]:bounding_box[1]+bounding_box[3]]
    # cv2.imwrite(image_path, crop_img)


def check_image_contain_face(name, file, created):
    # print("Information of that face", name)
    filename = secure_filename(file.filename)
    trained_storage = path.join(app.config['trained'], name)
    filename_change = str(created)+str(filename)
    image_path = path.join(trained_storage, filename_change)
    np_path = image_path+str('.npy')
    # file.save(path.join(app.config['trained'], filename_change))
    print('image_path', image_path)
    file.save(image_path)
    print("File is allowed and will be saved in ", trained_storage)
    # face_path = str('face')+image_path
    # list_tuple = extract_face_from_image(image_path)

    face_image = face_recognition.load_image_file(image_path)
    face_location = face_recognition.face_locations(face_image, number_of_times_to_upsample=1)
    if len(face_location) == 0:
        return 1, filename_change

    face_image_encoding = face_recognition.face_encodings(face_image, known_face_locations=face_location)[0]

    save(np_path, face_image_encoding)

    return 2, filename_change


@app.route('/api/add_user', methods=['POST'])
def add_user():
    print(json.dumps({'ip': request.remote_addr}))
    # if(request.remote_addr != '125.235.4.59' and request.remote_addr != '127.0.0.1'):
    #     return error_handle(10, "Not allow")
    if (checkIP(request.remote_addr) == 0):
        return error_handle(2, "IP này không được phép gửi request", "BLOCK_REQUEST")

    created1 = int(time.time())
    file1 = request.files['file']
    name = request.form['name']

    created2 = int(time.time())
    file2 = request.files['file2']
    created3 = int(time.time())
    file3 = request.files['file3']
    # if os.path.exists(path.join(app.config['trained'], name)) == False:
    #     print('check dir', os.path.exists(path.join(app.config['trained'], name)))
    #     os.mkdir(path.join(app.config['trained'], name))

    check_exist = check_user_is_exist(name)
    if check_exist == 1:
        remove_path_image(name)
        delete_user_by_name(name)
    else:
        if os.path.exists(path.join(app.config['trained'], name)) == False:
            os.mkdir(path.join(app.config['trained'], name))

    print('Check file is image', check_request_containt_image_file(request.files))
    print('request file', request.files)

    flag_check = check_request_containt_image_file(request.files)
    if(flag_check == 1):
        print("Not file in request")
        return error_handle(2, "Không có file trong request", "NOT_FOUND_FILE")
    if(flag_check == 2):
        print("File extension is not allowed")
        return error_handle(2, "Chỉ được upload file theo dạng .png và .jpg", "ERROR_FORMAT_FILE")

    print("Information of that face", name)

    flag_check_image_contain_face, filename_change1 = check_image_contain_face(name, file1, created1)
    if flag_check_image_contain_face == 1:
        remove_path_image(name)
        return error_handle(2, "Không tìm thấy khuôn mặt trong bức ảnh thứ nhất, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    flag_check_image_contain_face2, filename_change2 = check_image_contain_face(name, file2, created2)
    if flag_check_image_contain_face2 == 1:
        remove_path_image(name)

        return error_handle(2, "Không tìm thấy khuôn mặt trong bức ảnh thứ hai, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    flag_check_image_contain_face3, filename_change3 = check_image_contain_face(name, file3, created3)
    if flag_check_image_contain_face3 == 1:
        remove_path_image(name)

        return error_handle(2, "Không tìm thấy khuôn mặt trong bức ảnh thứ ba, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    try:

        # let start save file to our storage

        # save to our sqlite database.db

        user_id = app.db.insert('INSERT INTO users(name, created) values(%s,%s)', [name, created1])
        print("user has been saved")

        face_id = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename_change1, created1])
        face_id2 = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename_change2, created2])
        face_id3 = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename_change3, created3])

        print("face has been saved")
        # face_data = [{"id": face_id, "file_name": filename_change1, "created": created1},
        #              {"id": face_id2, "file_name": filename_change2, "created": created2},
        #              {"id": face_id3, "file_name": filename_change3, "created": created3}]
        app.face.update_model()
        return(success_handle(1, "Đã nhận được khuôn mặt", "VALID"))
    except Exception as e:
        print(e)
        return error_handle(0, "Lỗi khi thêm user vào DB", "ERROR_DB")
    # return success_handle(output)


def check_image_contain_face_add_url(name, url, created):
    page = requests.get(url)
    f_ext = os.path.splitext(url)[-1]
    trained_storage = path.join(app.config['trained'], name)
    filename = str(created)+str(f_ext)
    image_path = path.join(trained_storage, filename)
    with open(image_path, 'wb') as f:
        f.write(page.content)

    np_path = image_path+str('.npy')
    print("File is allowed and will be saved in ", trained_storage)

    face_image = face_recognition.load_image_file(image_path)

    face_location = face_recognition.face_locations(face_image, number_of_times_to_upsample=1)
    if len(face_location) == 0:
        return 1, filename

    face_image_encoding = face_recognition.face_encodings(face_image, known_face_locations=face_location, num_jitters=1)[0]

    save(np_path, face_image_encoding)

    return 2, filename


def check_url_user(url, created):
    page = requests.get(url)
    f_ext = os.path.splitext(url)[-1]
    print(f_ext)
    image_path = path.join(app.config['storage'], 'check_user', str(created)+str(f_ext))
    with open(image_path, 'wb') as f:
        f.write(page.content)

    print("File is allowed and will be saved in ", image_path)

    face_image = face_recognition.load_image_file(image_path)
    face_location = face_recognition.face_locations(face_image, number_of_times_to_upsample=1)
    if len(face_location) == 0:
        return 1

    return 2


@app.route('/api/check_user', methods=['POST'])
def check_user():
    created1 = int(time.time())
    # name = request.form['name']
    url1 = request.form['file']

    if check_url_is_image(url1) == 1:
        return error_handle(2, "URL không chứa ảnh", "URL_INVALID")

    if os.path.exists(path.join(app.config['storage'], 'check_user')) == False:
        os.mkdir(path.join(app.config['storage'], 'check_user'))

    if check_url_user(url1, created1) == 1:

        return error_handle(2, "Không tìm thấy khuôn mặt, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    return success_handle(1, "Đã nhận được khuôn mặt", "VALID")


@app.route('/api/add_url_user', methods=['POST'])
def add_url_user():
    print(json.dumps({'ip': request.remote_addr}))
    if (checkIP(request.remote_addr) == 0):
        return error_handle(2, "IP này không được phép gửi request", "BLOCK_REQUEST")

    created1 = int(time.time())
    created2 = created1+1
    created3 = created1+2

    name = request.form['name']
    url1 = request.form['file']
    url2 = request.form['file2']
    url3 = request.form['file3']
    print("Information of that face", name)
    if os.path.exists(path.join(app.config['trained'], name)) == False:
        os.mkdir(path.join(app.config['trained'], name))

    check_exist = check_user_is_exist(name)
    if check_exist == 1:
        remove_path_image(name)
        delete_user_by_name(name)
        print("Delete user completely")
        # return error_handle(2, "User is exist, you should change username")

    if check_url_is_image(url1) == 1:
        return error_handle(2, "URL không chứa ảnh", "URL_INVALID")
    if check_url_is_image(url2) == 1:
        return error_handle(2, "URL không chứa ảnh", "URL_INVALID")
    if check_url_is_image(url3) == 1:
        return error_handle(2, "URL không chứa ảnh", "URL_INVALID")

    flag_check_image_contain_face_add_url, filename1 = check_image_contain_face_add_url(name, url1, created1)
    if flag_check_image_contain_face_add_url == 1:
        remove_path_image(name)
        return error_handle(2, "Không tìm thấy khuôn mặt trong bức ảnh thứ nhất, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    flag_check_image_contain_face_add_url2, filename2 = check_image_contain_face_add_url(name, url2, created2)
    if flag_check_image_contain_face_add_url2 == 1:
        remove_path_image(name)
        return error_handle(2, "Không tìm thấy khuôn mặt trong bức ảnh thứ hai, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    flag_check_image_contain_face_add_url3, filename3 = check_image_contain_face_add_url(name, url3, created3)
    if flag_check_image_contain_face_add_url3 == 1:
        remove_path_image(name)
        return error_handle(2, "Không tìm thấy khuôn mặt trong bức ảnh thứ ba, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    try:

        # let start save file to our storage

        # save to our sqlite database.db

        user_id = app.db.insert('INSERT INTO users(name, created) values(%s,%s)', [name, created1])
        print("user has been saved")

        face_id = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename1, created1])
        face_id2 = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename2, created2])
        face_id3 = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename3, created3])

        print("face has been saved")
        # face_data = [{"id": face_id, "file_name": filename1, "created": created1},
        #              {"id": face_id2, "file_name": filename2, "created": created2},
        #              {"id": face_id3, "file_name": filename3, "created": created3}]
        app.face.update_model()
        return(success_handle(1, "Đã nhận được khuôn mặt", "VALID"))

    except Exception as e:
        print(e)
        return error_handle(0, "Lỗi khi thêm user vào DB", "ERROR_DB")

# route for user profile
# @app.route('/api/users/<string:name>', methods=['GET', 'DELETE'])
@app.route('/api/users', methods=['GET', 'DELETE'])
def user_profile():
    name = request.args.get('name')

    if request.method == 'GET':
        try:

            user = get_user_by_name(name)
            filename = user['faces'][0]['filename']
            path_image = path.join(app.config['trained'], name, filename)
            return send_file(path_image, mimetype='image/jpg')
        except Exception as e:
            print(e)
            return error_handle(4, "Không tìm thấy user trong DB", "USER_NOT_FOUND")

    if request.method == 'DELETE':
        try:
            remove_path_image(name)
            delete_user_by_name(name)
            return success_handle(1, "Xóa user thành công", "SUCCESSFULLY")

        except Exception as e:
            print(e)
            return error_handle(4, "Không tìm thấy user trong DB", "USER_NOT_FOUND")


# route for not found path image in storage
@app.route('/api/remove_if_not_in_storage', methods=['GET', 'DELETE'])
def users_not_path():
    name = request.args.get('name')

    if request.method == 'GET':
        try:
            user = get_user_by_name(name)
            return user
        except Exception as e:
            print(e)

            return error_handle(4, "Không tìm thấy user trong DB", "USER_NOT_FOUND")
    if request.method == 'DELETE':
        try:
            delete_user_by_name(name)
            return success_handle(1, "Xóa user thành công", "SUCCESSFULLY")

        except Exception as e:
            print(e)

            return error_handle(4, "Không tìm thấy user trong DB", "USER_NOT_FOUND")


# router for recognize a unknown face
@app.route('/api/recognize', methods=['POST'])
def recognize():
    print(json.dumps({'ip': request.remote_addr}))
    if (checkIP(request.remote_addr) == 0):
        return error_handle(2, "IP này không được phép gửi request", "BLOCK_REQUEST")

    # get file from requestcrop_img
    file = request.files['file']

    # get name in form datacrop_img
    name = request.form['name']
    created = int(time.time())

    check_exist = check_user_is_exist(name)
    if check_exist == 0:
        return error_handle(2, "Không tìm thấy ảnh người dùng trong DB", "NOT_FOUND_USERS")

    # check file is image
    # flag_check = check_request_containt_image_file(request.files)
    # if(flag_check == 1):
    #     print("Not file in request")
    #     return error_handle(10, "Not file in request")
    # if(flag_check == 2):
    #     print("File extension is not allowed")
    #     return error_handle(10, "We are only allow upload file with *.png , *.jpg")

    if os.path.exists(path.join(app.config['unknown'], name)) == False:
        os.mkdir(path.join(app.config['unknown'], name))

    print("Information of that image", name)
    filename = secure_filename(file.filename)
    unknown_storage = path.join(app.config['storage'], 'unknown', name)
    unknown_image_path = path.join(unknown_storage, filename)
    file.save(unknown_image_path)
    print("File is allowed and will be saved in ", unknown_storage)

    # encoding unknown image
    unknown_face_image = face_recognition.load_image_file(unknown_image_path)
    unknown_face_location = face_recognition.face_locations(unknown_face_image, number_of_times_to_upsample=1)
    if len(unknown_face_location) == 0:
        os.remove(unknown_image_path)
        return error_handle(2, "Không tìm thấy khuôn mặt trong bức ảnh, xin vui lòng thử lại ảnh khác", "NOT_FOUND_FACE")

    if(len(unknown_face_location) > 1):
        print(unknown_face_location)
        return success_handle(2, 'Phát hiện gian lận', 'CHEAT')
    print("found face in image")

    try:
        app.face.load_all()
        app.face.init_index()

        output = app.face.recognize2(name, unknown_face_location, unknown_face_image)

        # os.remove(unknown_image_path)
        return success_handle(output['code'], output['message'], output['status'])

    except Exception as e:
        # os.remove(unknown_image_path)
        print(e)
        return error_handle(2, "Không tìm thấy ảnh người dùng trong DB", "NOT_FOUND_USERS")

    return success_handle(1, "Hợp lệ", "VALID")


@app.route('/api/live_recognition', methods=['POST'])
def live_recognition():
    app.live_face = Live_Face(app)

    app.live_face.live_recognize()
    return success_handle(1, "Test thành công", "TEST_SUCCESS")


# Run the app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
