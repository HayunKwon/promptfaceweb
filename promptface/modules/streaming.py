"""
This module is a modified version of deepface.modules.streaming.py

get_camera: get camera from SOURCE
process: modify img and return img, target_path, target_distance
"""

# built-in dependencies
import math
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

# 3rd party dependencies
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import cv2
from sklearn.metrics.pairwise import cosine_distances

# deepface
from deepface import DeepFace
from deepface.modules.verification import find_threshold
from deepface.modules.streaming import IDENTIFIED_IMG_SIZE
from deepface.modules.streaming import (
    overlay_identified_face,
    highlight_facial_areas,
    countdown_to_freeze,
    countdown_to_release,
    grab_facial_areas,
)

# project dependencies
from promptface.utils.constants import *
from promptface.utils.logger import Logger

logger = Logger(__name__)

def get_camera():
    """
    get cv2.VideoCapture safely

    Returns:
        capture (cv2.VideoCapture): 
    """
    cap = cv2.VideoCapture(SOURCE)

    has_frame, _ = cap.read()
    if has_frame is None:
        raise ValueError('No Camera')

    return cap


class AbstractPromptface(ABC):
    def __init__(self):
        self.num_frames_with_faces:int = 0
        self.freezed_img:Optional[cv2.typing.MatLike] = None
        self.freeze:bool = False
        self.tic:float = time.time()
        self.target_path:Optional[str] = None
        self.target_distance:Optional[NDArray[np.float64]] = None

        if RASPI_IP == None:
            self.cap = get_camera()
            _, img = self.cap.read()
            size = img.shape[0] * img.shape[1]
            self.threshold_size = int(math.sqrt(size*DISCARD_PERCENTAGE/100))


    @classmethod
    @abstractmethod
    def app(cls, callback, *args, **kwargs):
        pass


    def process(
            self,
            img: cv2.typing.MatLike,
            database_embeddings: NDArray[np.float64],
            identities: pd.Series,
            face_overlay: bool = True
        ):
        """
        modify img and return img, target_path, target_distance

        Args:
            img (MatLike): image from VideoCapture
            database_embeddings (NDArray[np.float64]): vectorized embeddings from database
            identities (Series): Coloum from df
        Returns:
            img (MatLike): freezed and countdown img, raw img, etc.. processed img

        Static Variables (attribute):
            num_frames_with_faces (int) = how many sequantial frames do we have with face(s)

            freezed_img (MatLike): when face detected in cap, freeze img and store to release same img

            freeze (bool): when face detected in cap, freeze

            tic (float): start time during last freeze or start main

            target_path (str): save target_path, determine if it is verify in main

            target_distance (NDArray): save target_distance, do `something` with target_path or target_distance in main, you SHOULD RESET None.
                o/w, you will continue to do `something` as much as possible in freezing time
        """
        raw_img = img.copy()

        faces_coordinates = []
        if self.freeze is False:
            # [0:1] means get first face.
            # threshold means discard threshold x threshold size face
            # TODO DISCARD_PERCENTAGE to THRESHOLD
            faces_coordinates = grab_facial_areas(img=img, detector_backend=DETECTOR_BACKEND, threshold=self.threshold_size)[0:1]

            # we will pass img to analyze modules (identity, demography) and add some illustrations
            # that is why, we will not be able to extract detected face from img clearly

            img = highlight_facial_areas(img=img, faces_coordinates=faces_coordinates)
            img = countdown_to_freeze(img=img,
                                    faces_coordinates=faces_coordinates,
                                    frame_threshold=FRAME_THRESHOLD,
                                    num_frames_with_faces=self.num_frames_with_faces,)

            self.num_frames_with_faces = self.num_frames_with_faces + 1 if len(faces_coordinates) else 0

            self.freeze = self.num_frames_with_faces > 0 and self.num_frames_with_faces % FRAME_THRESHOLD == 0
            if self.freeze:
                # add analyze results into img - derive from raw_img
                img = highlight_facial_areas(img=raw_img, faces_coordinates=faces_coordinates)

                # if verify, we can get target_path and target_distance
                # perfom_facial_recognition is customed
                img = self.perform_facial_recognition(img=img,
                                                      faces_coordinates=faces_coordinates,
                                                      database_embeddings=database_embeddings,
                                                      df_identities=identities,
                                                      face_overlay=face_overlay,
                                                  )

                # # check img is None
                # assert self.target_path is not None, 'Verification failed'

                if self.freezed_img is None:
                    # freeze the img after analysis
                    self.freezed_img = img.copy()
                    # start counter for freezing
                    self.tic = time.time()
                    logger.info("freezed")

        elif self.freeze is True and time.time() - self.tic > TIME_THRESHOLD:
            # reset variables
            self.freeze = False
            self.freezed_img = None
            self.target_path = None
            self.target_distance = None
            # reset counter for freezing
            self.tic = time.time()
            logger.info("freeze released\n")

        self.freezed_img = countdown_to_release(img=self.freezed_img, tic=self.tic, time_threshold=TIME_THRESHOLD)
        return img if self.freezed_img is None else self.freezed_img


    # this function is a modified version of deepface.modules.streaming.perform_facial_recognition
    def perform_facial_recognition(
            self,
            img: cv2.typing.MatLike,
            # detected_faces: List[NDArray],
            faces_coordinates: List[Tuple[int, int, int, int]],
            database_embeddings: NDArray[np.float64],
            df_identities: pd.Series,
            face_overlay:bool = True,
        ):
        """
        Perform facial recognition, verification

        Args:
            img (np.ndarray): image itself
            faces_coordinates (list): list of facial area coordinates as tuple with
                x, y, w and h values
            database_embeddings (list): list of embedded database images
            df_identities (Series): series of embedded database images identities
        Returns:
            img (np.ndarray): image with identified face informations
        """
        is_verify = False

        for (x, y, w, h) in faces_coordinates:
            target_path = None

            # vectorize embedding
            embedding_objs = DeepFace.represent(img_path=img,
                                                model_name=MODEL_NAME,
                                                enforce_detection=ENFORCE_DETECTION,
                                                detector_backend=DETECTOR_BACKEND,
                                                align=ALIGN,)
            # TODO check what is the first face if detected face is more then one
            embedding:list = embedding_objs[0]["embedding"]
            new_embedding = np.array(embedding, dtype=np.float64)

            # cosine distance
            distance = cosine_distances(database_embeddings, new_embedding.reshape(1, -1))
            most_similar_index = np.argmin(distance)
            logger.debug(f'the most similar {most_similar_index} {df_identities[most_similar_index]} {distance[most_similar_index]}')

            # check same person
            # this part is changed 'search_identify->check target_label' to 'verify cos_distance->check verify'
            # TODO ArcFace, cosine gets 0.68, but i thing 0.68 is pretty spacious
            if distance[most_similar_index] >= find_threshold(model_name=MODEL_NAME, distance_metric='cosine'):
                continue

            # --------------------------------------------
            # RUN AFTER this part when verify at least one
            is_verify = True

            # get path
            target_path = df_identities[most_similar_index]
            logger.debug(f"Hello, {target_path} {distance[most_similar_index]}")

            # load found identity image - extracted if possible
            target_objs = DeepFace.extract_faces(
                img_path=target_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=ENFORCE_DETECTION,
                align=ALIGN,
            )

            # extract facial area of the identified image if and only if it has one face
            # otherwise, show image as is
            if len(target_objs) == 1:
                # extract 1st item directly
                target_obj = target_objs[0]
                target_img = target_obj["face"]
                target_img = cv2.resize(target_img, (IDENTIFIED_IMG_SIZE, IDENTIFIED_IMG_SIZE))
                target_img *= 255   # to set RGB as 0~255
                target_img = target_img[:, :, ::-1]

            else:
                target_img = cv2.imread(target_path)
                target_img = self.__padded_img(target_img)

            target_label = target_path.split('\\')[-1].split('.')[0]

            if face_overlay:
                img = overlay_identified_face(
                    img=img,
                    target_img=target_img,
                    label="{} {}".format(target_label, round(float(distance[most_similar_index]), 2)),
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                )

        # pass through func without changing img when is_verify is False
        # return img, target_path if is_verify else None, distance[most_similar_index]
        self.target_path = target_path if is_verify else None
        self.target_distance = distance[most_similar_index]
        return img


    def __padded_img(self, img:cv2.typing.MatLike):
        """
        get image from database and resize to (IDENTIFIED_IMG_SIZE, IDENTIFIED_IMG_SIZE, 3)

        Args:
            img (MatLike): specific img from database
        Returns:
            new_img (NDArray[unit8]): resized img
        """
        dim = self.__resized_dim(img.shape)
        new_img = np.zeros((IDENTIFIED_IMG_SIZE, IDENTIFIED_IMG_SIZE, 3), np.uint8)
        img = cv2.resize(img, dim)

        top = (IDENTIFIED_IMG_SIZE - dim[1]) // 2
        bottom = (IDENTIFIED_IMG_SIZE + dim[1]) // 2
        left = (IDENTIFIED_IMG_SIZE - dim[0]) // 2
        right = (IDENTIFIED_IMG_SIZE + dim[0]) // 2
        new_img[top:bottom, left:right] = img

        return new_img


    def __resized_dim(self, shape: cv2.typing.MatShape):
        """
        resize original shape to IDENTIFIED_IMG_SIZE

        Args:
            shape (MatShape): it contains height, weight, channels
        Returns:
            Tuple(resized_w, resized_h)
        """
        h, w, _ = shape

        if h < w:
            return IDENTIFIED_IMG_SIZE, round(h / (w / IDENTIFIED_IMG_SIZE)),
        if h > w:
            return round(w / (h / IDENTIFIED_IMG_SIZE)), IDENTIFIED_IMG_SIZE
        return IDENTIFIED_IMG_SIZE, IDENTIFIED_IMG_SIZE
