"""
This module is a modified version of deepface.commons.constant.py

DB_PATH: local db_path
MODEL_NAME: model like ArcFace, Facenet512, ...
DETECTOR_BACKEND: detector like opencv, yolov8, ...
ENFORCE_DETECTION: if this constant is True, you would get Error when detector can't detect faces
ALIGN: Perform alignment based on the eye positions.
DISCARD_PERCENTAGE: discard face img size per full img size ratio.
    If it is 2 percent of the 1280x720 resolution, it returns the threshold of 136x136.
SOURCE: camera number
TIME_THRESHOLD: freezing time
FRAME_THRESHOLD: ensure the face when face is detected during N frames
BROKER_IP: IP which is using for broker
BROKER_PORT: default port is 1883.
TOPIC_STREAM: "home/stream"
TOPIC_RESULT: "home/result"
INFO_FORMAT: stream format for log
"""
# built-in dependencies
import json
from typing import Dict, Any

# project dependencies
from promptface.utils.logger import Logger

logger = Logger(__name__)


_DEFAULT_DATA = {
    "DB_PATH": "./ImgDataBase",
    "MODEL_NAME": "ArcFace",
    "DETECTOR_BACKEND": "opencv",
    "ENFORCE_DETECTION": True,
    "ALIGN": True,
    "DISCARD_PERCENTAGE": 2,
    "SOURCE": 0,
    "TIME_THRESHOLD": 5,
    "FRAME_THRESHOLD": 10,
    "BROKER_IP": None,
    "BROKER_PORT": 1883,
    "TOPIC_STREAM": "home/stream",
    "TOPIC_RESULT": "home/result"
}


# load external constants and variables
try:
    with open('./constants.json', 'r') as f:
        data: Dict[str, Any] = json.load(f)
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
        "FRAME_THRESHOLD": 10,
        "BROKER_IP": None,
        "BROKER_PORT": 1883,
        "TOPIC_STREAM": "home/stream",
        "TOPIC_RESULT": "home/result"
    }
    with open('./constants.json', 'w') as f:
        json.dump(data, f, indent='\t')
    logger.info('success to create ./constants.json file')
finally:
    for key, value in data.items():
        if value == None:
            data[key] = _DEFAULT_DATA[key]
    logger.debug('load ./constants.json')


# db path
DB_PATH:str = data["DB_PATH"]


# about deepface
MODEL_NAME:str = data["MODEL_NAME"]
DETECTOR_BACKEND:str = data["DETECTOR_BACKEND"]
# if False, you'll have a full image embeddings as a face image, not None, when no detected face
ENFORCE_DETECTION:bool = data["ENFORCE_DETECTION"]
ALIGN:bool = data["ALIGN"]
DISCARD_PERCENTAGE:int = data["DISCARD_PERCENTAGE"]


# about cv2
SOURCE:int = data["SOURCE"]
TIME_THRESHOLD:int = data["TIME_THRESHOLD"]
FRAME_THRESHOLD:int = data["FRAME_THRESHOLD"]


# MQTT
BROKER_IP:str = data["BROKER_IP"]
BROKER_PORT:int = data["BROKER_PORT"]
TOPIC_STREAM:str = data["TOPIC_STREAM"]
TOPIC_RESULT:str = data["TOPIC_RESULT"]


# stream format
INFO_FORMAT = '----- {} -----'
