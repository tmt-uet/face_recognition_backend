from flask import Flask, json, Response, request, render_template, send_file
from os import path, getcwd
import os
import time
import face_recognition
import mysql.connector
import urllib
from urllib.request import urlopen
import io
import requests
import shutil
from flask_cors import CORS
from face import Live_Face
app = Flask(__name__)
app.face = Live_Face(app)



# Run the app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
