"""
This module is a modified version of deepface.commons.constant.py

DB_PATH: local db_path
MODEL_NAME: model like ArcFace, Facenet512, ...
DETECTOR_BACKEND: detector like opencv, yolov8, ...
ENFORCE_DETECTION: if this constant is True, you would get Error when detector can't detect faces
ALIGN: Perform alignment based on the eye positions.
DISCARD_PERCENTAGE: discard face img size per full img size ratio.
SOURCE: camera number
TIME_THRESHOLD: freezing time
FRAME_THRESHOLD: ensure the face when face is detected during N frames
INFO_FORMAT: stream format for log
"""
# built-in dependencies
import json

# project dependencies
from promptface.utils.logger import Logger

logger = Logger(__name__)


# load external constants and variables
try:
    with open('./constants.json', 'r') as f:
        data:dict = json.load(f)
except FileNotFoundError as e:
    logger.error(str(e))
    data = {
        "DB_PATH": "./ImgDataBase",
        "MODEL_NAME": None,
        "DETECTOR_BACKEND": None,
        "ENFORCE_DETECTION": True,
        "ALIGN": True,
        "DISCARD_PERCENTAGE": 2,
        "SOURCE": 0,
        "TIME_THRESHOLD": 5,
        "FRAME_THRESHOLD": 10
    }
    with open('./constants.json', 'w') as f:
        json.dump(data, f, indent='\t')
    logger.info('success to create ./constants.json file')
logger.debug('load ./constants.json')

# db path
DB_PATH = data["DB_PATH"] if data["DB_PATH"] else './ImgDataBase'


# about deepface
MODEL_NAME = data["MODEL_NAME"] if data["MODEL_NAME"] else 'ArcFace'
DETECTOR_BACKEND = data["DETECTOR_BACKEND"] if data["DETECTOR_BACKEND"] else 'opencv'
# IMPORTANT
# if False, you'll have a full image embeddings as a face image, not None, when no detected face
ENFORCE_DETECTION = data["ENFORCE_DETECTION"] if data["ENFORCE_DETECTION"] else True
ALIGN = data["ALIGN"] if data["ALIGN"] else True
DISCARD_PERCENTAGE = data["DISCARD_PERCENTAGE"] if data["DISCARD_PERCENTAGE"] else 2


# about cv2
SOURCE = data["SOURCE"] if data["SOURCE"] else 0
TIME_THRESHOLD = data["TIME_THRESHOLD"] if data["TIME_THRESHOLD"] else 5
FRAME_THRESHOLD = data["FRAME_THRESHOLD"] if data["FRAME_THRESHOLD"] else 5


# stream format
INFO_FORMAT = '----- {} -----'
