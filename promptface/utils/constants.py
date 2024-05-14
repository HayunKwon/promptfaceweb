"""
This module is a modified version of deepface.commons.constant.py

DB_PATH: local db_path
MODEL_NAME: model like ArcFace, Facenet512, ...
DETECTOR_BACKEND: detector like opencv, yolov8, ...
ENFORCE_DETECTION: if this constant is True, you would get Error when detector can't detect faces
ALIGN: Perform alignment based on the eye positions.
SOURCE: camera number
TIME_THRESHOLD: freezing time
FRAME_THRESHOLD: ensure the face when face is detected during N frames
INFO_FORMAT: stream format for log
"""
import json


# load external constants and variables
with open('./constants.json') as f:
    data:dict = json.load(f)


# db path
DB_PATH = data["DB_PATH"] if data["DB_PATH"] else './ImgDataBase'


# about deepface
MODEL_NAME = data["MODEL_NAME"] if data["MODEL_NAME"] else 'ArcFace'
DETECTOR_BACKEND = data["DETECTOR_BACKEND"] if data["DETECTOR_BACKEND"] else 'opencv'
# IMPORTANT
# if False, you'll have a full image embeddings as a face image, not None, when no detected face
ENFORCE_DETECTION = data["ENFORCE_DETECTION"] if data["ENFORCE_DETECTION"] else True
ALIGN = data["ALIGN"] if data["ALIGN"] else True


# about cv2
SOURCE = data["SOURCE"] if data["SOURCE"] else 0
TIME_THRESHOLD = data["TIME_THRESHOLD"] if data["TIME_THRESHOLD"] else 5
FRAME_THRESHOLD = data["FRAME_THRESHOLD"] if data["FRAME_THRESHOLD"] else 5


# stream format
INFO_FORMAT = '----- {} -----'
