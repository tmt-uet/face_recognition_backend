from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import os
import time
from my_sql_db import Database
# from face import Face
import face_recognition
import mysql.connector

# from gw_utility.logging import Logging


app = Flask(__name__)
app.config['file_allowed'] = ['image/png', 'image/jpeg']
app.config['storage'] = path.join(getcwd(), 'storage')
app.db = Database()
# app.face = Face(app)


def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({"error": {"message": error_message}}), status=status, mimetype=mimetype)


def get_user_by_id(user_id):
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


def remove_path_image(user_id):
    results = app.db.select('SELECT filename FROM faces WHERE faces.user_id= %s', [user_id])
    print("errrrrrrrrrrrrroorrrrrrrrrrrrrrrrr")
    remove_path = path.join(app.config['storage'], 'trained', results[0][0])
    os.remove(remove_path)


def delete_user_by_id(user_id):
    app.db.delete('DELETE FROM users WHERE users.id = %s', [user_id])
    # also delete all faces with user id
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

            print("File is allowed and will be saved in ", app.config['storage'])
            filename = secure_filename(file.filename)
            trained_storage = path.join(app.config['storage'], 'trained')
            image_path = path.join(trained_storage, filename)
            file.save(image_path)

            face_image = face_recognition.load_image_file(image_path)
            try:
                face_image_encoding = face_recognition.face_encodings(face_image)[0]
                print("found face in image")

                # let start save file to our storage

                # save to our sqlite database.db
                created = int(time.time())
                user_id = app.db.insert('INSERT INTO users(name, created) values(%s,%s)', [name, created])
                if user_id:
                    print("User saved in database", name, user_id)

                    # user has been save with user_id and now we need save faces table
                    face_id = app.db.insert('INSERT INTO faces(user_id, filename, created) values(%s,%s,%s)', [user_id, filename, created])
                    if face_id:
                        print("face has been saved")
                        face_data = {"id": face_id, "file_name": filename, "created": created}
                        return_output = json.dumps({"id": user_id, "name": name, "face": [face_data]})
                        return(success_handle(return_output))
                    else:
                        print("An error saving face image")
                        return(error_handle("An error saving face image"))
                else:
                    print("Something happend")
                    return error_handle("An error inserting new user")

            except:
                os.remove(image_path)
                # print("not found face in image")
                # output = json.dumps({"error": "Not found face in an image, try other images"})
                return(error_handle("Not found face in an image, try other images"))

            return success_handle(output)


# route for user profile
@app.route('/api/users/<int:user_id>', methods=['GET', 'DELETE'])
def user_profile(user_id):
    if request.method == 'GET':
        user = get_user_by_id(user_id)
        if user:
            return success_handle(json.dumps(user))
        else:
            return error_handle("User not found")
    if request.method == 'DELETE':
        try:
            remove_path_image(user_id)
        except:
            return error_handle("Not found result in database")

        delete_user_by_id(user_id)

        return success_handle(json.dumps({"deleted": True}))

# router for recognize a unknown face
# @app.route('/api/recognize', methods=['POST'])
# def recognize():
#     if 'file' not in request.files:
#         return error_handle("Image is required")
#     else:
#         file = request.files['file']
#         # file extension valiate
#         if file.mimetype not in app.config['file_allowed']:
#             return error_handle("File extension is not allowed")
#         else:

#             filename = secure_filename(file.filename)
#             unknown_storage = path.join(app.config["storage"], 'unknown')
#             file_path = path.join(unknown_storage, filename)
#             file.save(file_path)

#             user_id = app.face.recognize(filename)
#             if user_id:
#                 user = get_user_by_id(user_id)
#                 message = {"message": "Hey we found {0} matched with your face image".format(user["name"]),
#                            "user": user}
#                 return success_handle(json.dumps(message))
#             else:

#                 return error_handle("Sorry we can not found any people matched with your face image, try another image")


# Run the app

if __name__ == '__main__':
    app.run()
