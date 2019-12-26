from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import os
import time
from my_sql_db import Database
from face_recog import Face
import face_recognition
import mysql.connector

# from gw_utility.logging import Logging


app = Flask(__name__)
app.config['file_allowed'] = ['image/png', 'image/jpeg']
app.config['storage'] = path.join(getcwd(), 'storage')
app.db = Database()
app.face = Face(app)


def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({"error": {"message": error_message}}), status=status, mimetype=mimetype)


def get_user_by_name(name):
    print("get_user_by_name")
    user_id = app.db.select('SELECT id from users WHERE name= %s', [name])[0][0]
    user = {}
    results = app.db.select(
        'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.id = %s',
        [user_id])

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
    user_id = app.db.select('SELECT id from users WHERE name= %s', [name])[0][0]

    results = app.db.select('SELECT filename FROM faces WHERE faces.user_id= %s', [user_id])
    # print("errrrrrrrrrrrrroorrrrrrrrrrrrrrrrr")
    remove_path = path.join(app.config['storage'], 'trained', results[0][0])
    os.remove(remove_path)


def delete_user_by_name(name):
    # print(name)
    user_id = app.db.select('SELECT id from users WHERE name= %s', [name])[0][0]
    print(user_id)
    app.db.delete('DELETE FROM users WHERE users.id = %s', [user_id])

    app.db.delete('DELETE FROM faces WHERE faces.user_id = %s', [user_id])


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api', methods=['GET'])
def homepage():
    output = json.dumps({"api": '1.0'})
    return success_handle(output)


@app.route('/api/train', methods=['POST'])
def train():
    output = json.dumps({"success": True})

    if 'file' not in request.files:

        print("Not file in request")
        return error_handle("Not file in request")
    else:

        print("File request", request.files)
        file = request.files['file']

        if file.mimetype not in app.config['file_allowed']:

            print("File extension is not allowed")

            return error_handle("We are only allow upload file with *.png , *.jpg")
        else:

            # get name in form data
            name = request.form['name']

            print("Information of that face", name)
            filename = secure_filename(file.filename)
            trained_storage = path.join(app.config['storage'], 'trained')
            image_path = path.join(trained_storage, filename)
            file.save(image_path)

            print("File is allowed and will be saved in ", trained_storage)

            face_image = face_recognition.load_image_file(image_path)
            try:
                face_image_encoding = face_recognition.face_encodings(face_image)[0]
                print("found face in image")

            except:
                os.remove(image_path)
                # print("not found face in image")
                # output = json.dumps({"error": "Not found face in an image, try other images"})
                return(error_handle("Not found face in an image, try other images"))
            try:

                # let start save file to our storage

                # save to our sqlite database.db
                created = int(time.time())
                user_id = app.db.insert('INSERT INTO users(name, created) values(%s,%s)', [name, created])
                print("user has been saved")

                face_id = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename, created])
                print("face has been saved")
                face_data = {"id": face_id, "file_name": filename, "created": created}
                return_output = json.dumps({"id": user_id, "name": name, "face": [face_data]})

                return(success_handle(return_output))
            except Exception as e:
                print(e)
                return error_handle("ERRORRRRRRRRRRRR")
            return success_handle(output)


# route for user profile
@app.route('/api/users/<string:name>', methods=['GET', 'DELETE'])
def user_profile(name):
    if request.method == 'GET':
        try:
            user = get_user_by_name(name)
            return user
        except Exception as e:
            print(e)
            return error_handle("User not found")

    if request.method == 'DELETE':
        try:
            remove_path_image(name)
            delete_user_by_name(name)
            return success_handle(json.dumps({"deleted": True}))

        except Exception as e:
            print(e)
            return error_handle("Not found result in database or not found image")


# route for not found path image in storage
@app.route('/api/remove_if_not_in_storage/<string:name>', methods=['GET', 'DELETE'])
def users_not_path(name):
    if request.method == 'GET':
        try:
            user = get_user_by_name(name)
            return user
        except:
            return error_handle("User not found")
    if request.method == 'DELETE':
        try:
            delete_user_by_name(name)
            return success_handle(json.dumps({"deleted": True}))

        except:
            return error_handle("Not found result in database")


# router for recognize a unknown face
@app.route('/api/recognize', methods=['POST'])
def recognize():
    output = json.dumps({"success": True})

    if 'file' not in request.files:

        print("Not file in request")
        return error_handle("Not file in request")
    else:

        print("File request", request.files)
        file = request.files['file']

        if file.mimetype not in app.config['file_allowed']:

            print("File extension is not allowed")

            return error_handle("We are only allow upload file with *.png , *.jpg")
        else:

            # get name in form data
            name = request.form['name']

            print("Information of that image", name)
            filename = secure_filename(file.filename)
            unknown_storage = path.join(app.config['storage'], 'unknown')
            unknown_image_path = path.join(unknown_storage, filename)
            file.save(unknown_image_path)
            print("File is allowed and will be saved in ", unknown_storage)

            face_image = face_recognition.load_image_file(unknown_image_path)
            try:
                face_image_encoding = face_recognition.face_encodings(face_image)[0]
                print("found face in image")

                # return success_handle(output)
            except Exception as e:
                print(e)
                os.remove(unknown_image_path)
                returnerror_handle("Not found face in an image, try other images")

            try:
                confirm = app.face.recognize(name, unknown_image_path)
                if confirm == True:
                    return success_handle(json.dumps({"message": "Valid image"}))
                else:
                    return success_handle(json.dumps({"message": "Invalid image"}))
            except Exception as e:
                print(e)
                return error_handle("Not found image of account in database")

            return success_handle(output)
            # if user_id:
            #     user = get_user_by_id(user_id)
            #     message = {"message": "Hey we found {0} matched with your face image".format(user["name"]),
            #                "user": user}
            #     return success_handle(json.dumps(message))
            # else:

            #     return error_handle("Sorry we can not found any people matched with your face image, try another image")


# Run the app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
