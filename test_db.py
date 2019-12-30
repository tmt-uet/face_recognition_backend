import face_recognition
from os import path
import numpy as np
import requests
from my_sql_db import Database
db = Database()
name = 'Tung'
# results = db.select(
#     'SELECT users.id, users.name, users.created, faces.id, faces.user_id, faces.filename,faces.created FROM users LEFT JOIN faces ON faces.user_id = users.id WHERE users.name = %s',
#     [name])
# re = db.select('SELECT id from users WHERE name= %s', [name])[0][0]
# # print(results[0][5])
# print(results)
check_exist = db.select('SELECT count(*) from users WHERE name=%s', [name])
if (len(check_exist)) > 0:
    print("YES")
print(len(check_exist))
