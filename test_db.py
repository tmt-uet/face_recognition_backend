import face_recognition
from os import path
import numpy as np
import requests
from my_sql_db import Database
# db = Database()
# name = 'Trang'
# results = db.select(
#     'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.name = %s',
#     ['Trang'])
# re = db.select('SELECT id from users WHERE name= %s', [name])[0][0]
# print(results[0][5])

import base64
from PIL import Image
from io import BytesIO
with open("/home/tmt/Documents/face_recognition/my_app/storage/trained/2019-10-30.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
print(encoded_string)

im = Image.open(BytesIO(base64.b64decode(encoded_string)))
im.save('/home/tmt/Documents/face_recognition/my_app/image1.png', 'PNG')
