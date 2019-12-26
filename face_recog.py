import face_recognition
from os import path
import numpy as np
import requests


class Face:
    def __init__(self, app):
        self.storage = app.config["storage"]
        self.db = app.db

    def get_path_image_in_db(self, name):

    def recognize(self, name, image_request):
