"""
This module is a modified version of deepface.modules.recognition.py

show_pkl: log pkl df and show cropped face img
load_pkl: init pkl and return vectorized embeddings, identities
init_pkl: init pkl and update pkl
"""

# built-in dependencies
import os
import copy
import time
import pickle
import operator
from typing import List, Dict, Any

# 3rd part dependencies
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm   # for __find_buld_embeddings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# deepface
from deepface.commons import image_utils
from deepface.modules import detection, representation

# project dependencies
from promptface.utils.constants import ALIGN, DB_PATH, DETECTOR_BACKEND, ENFORCE_DETECTION, INFO_FORMAT, MODEL_NAME
from promptface.utils.logger import Logger

logger = Logger(__name__)


def show_pkl(show_plt=False):
    """
    Log pkl that full imgs, face cropped imgs and ratio of face area

    Args:
        show_plt (bool): if True, show plt.
    """
    # ----- INIT -----
    try:
        # set logger
        show_logger = Logger(__name__, 'Logs/show_pickel.log')
        show_logger.info(INFO_FORMAT.format('APP START'))

        # init pickle and get representatinos
        representations = init_pkl()
    except Exception as e:
        show_logger.critical(str(e))
        exit(1)


    # ----- MAIN -----
    show_logger.info(INFO_FORMAT.format('SHOW PICKLE'))
    df = pd.DataFrame(representations)
    show_logger.info('\n{}'.format(df))

    # img_paths = representations.identity
    for _, data_row in df.iterrows():
        # set data
        path = data_row.identity
        is_face = True if data_row.embedding else False
        show_logger.info('{}\tface area ratio: {}%'.format(path, round((data_row.target_w * data_row.target_h) / (data_row.original_shape[0] * data_row.original_shape[1]) * 100, 2)))

        if show_plt is False:
            continue

        # set img
        original_img = cv2.imread(path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        face_img = np.zeros((original_img.shape), np.uint8)
        if is_face:
            left, top = data_row.target_x, data_row.target_y
            right, bottom = left + data_row.target_w, top + data_row.target_h
            face_img = copy.deepcopy(original_img[top:bottom, left:right])
            thickness = original_img.shape[1] // 350
            original_img = cv2.rectangle(original_img, (left, top), (right, bottom), (255, 67, 67), thickness)


        # save result
        k = 1.2
        fig_img = plt.figure(figsize=(8*k, 5*k))
        gs = GridSpec(ncols=2, nrows=1, figure=fig_img)

        ax0 = fig_img.add_subplot(gs[: ,0])
        ax0.title.set_text('Original image')
        ax0.imshow(original_img)

        ax1 = fig_img.add_subplot(gs[0, 1])
        ax1.title.set_text('detect: {}'.format(is_face))
        ax1.imshow(face_img, interpolation='none')
        ax1.axis('off')

        # show
        fig_img.suptitle("{} {}\nx, y, w, h\n{} {} {} {}".format(path, is_face, data_row.target_x, data_row.target_y, data_row.target_w, data_row.target_h))
        fig_img.tight_layout()
        plt.show()
        plt.close()


# ---------------
# custom load pkl
def load_pkl(db_path = DB_PATH):
    """
    return vectorized embedings from ImgDB and identities from df after init pkl

    Args:
        db_path (str): image database path
    Returns:
        vectorized embeddings (NDArray[np.float64]): vertorized df.embeddings
        identities (Series): Series from df.identities
    """
    representation = init_pkl(db_path=db_path)
    logger.info('complete to initialize pickle')

    embeddings, identities = __get_embeddings_from_db(representation)
    
    return __vectorize(embeddings=embeddings), identities


def __vectorize(embeddings: pd.Series):
    """
    vectorize df.embeddings

    Args:
        embeddings (Series): embeddings from df
    Returns:    
        embeddings (NDArray[np.float64]): vectorized embeddings from df
    """
    database_embeddings = embeddings.to_list()
    database_embeddings = np.array(database_embeddings, dtype=np.float64)
    return database_embeddings


def __get_embeddings_from_db(representations:List[Dict[str, Any]]):
    """
    get embeddings from db without embedding is None

    Args:
        representations (list): list of df cols
    Returns:
        embeddings (Series): df col of embedding without missing values
        identities (Series): df col of identities without missing values
    """
    # get DataFrames from representations
    df_all = pd.DataFrame(representations)
    df_null = df_all.embedding.isnull()
    df = df_all.dropna()

    logger.info('get representations from database'
                f', {len(df)} detected'
                f', {df_null.sum()} is None'
            )

    # if missing values
    if df_null.sum():
        logger.warning(f'None objects\n{df_all[df_null]}')
    
    # check df
    assert isinstance(df, pd.DataFrame), 'dataframe is None'
    
    embeddings = df.embedding
    identities = df.identity

    # 재정렬
    embeddings.index = [i for i in range(len(embeddings.index))]
    identities.index = [i for i in range(len(identities.index))]
    return embeddings, identities


# ------------------------------------------------------------
# this function is a part of deepface.modules.recognition.find
def init_pkl(
        db_path: str = DB_PATH,
        model_name: str = MODEL_NAME,
        enforce_detection: bool = ENFORCE_DETECTION,
        detector_backend: str = DETECTOR_BACKEND,
        align: bool = ALIGN,
        expand_percentage: int = 0,
        normalization: str = "base",
        silent: bool = False,
    ) -> List[Dict["str", Any]]:
    """
    init pkl and update pkl

    Args:
        db_path (string): Path to the folder containing image files. All detected faces
            in the database will be considered in the decision-making process.

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is ArcFace).

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

        silent (boolean): Suppress or allow some log messages for a quieter analysis process.
    Returns:
        representations (List[Dict["str", Any]]): list of pd.DataFrame

            identity: path of img

            hash: img hash

            embedding: embedding of target (= face)

            original_shape: shape of identity img
            
            'target_x', 'target_y', 'target_w', 'target_h': bounding box coordinates of the
                    target face in the database.
    """
    tic = time.time()

    # -------------------------------
    if os.path.isdir(db_path) is not True:
        raise ValueError("Passed db_path does not exist!")

    # ---------------------------------------

    file_parts = [
        "ds",
        "model",
        model_name,
        "detector",
        detector_backend,
        "aligned" if align else "unaligned",
        "normalization",
        normalization,
        "expand",
        str(expand_percentage),
    ]

    file_name = "_".join(file_parts) + ".pkl"
    file_name = file_name.replace("-", "").lower()

    datastore_path = os.path.join(db_path, file_name)
    representations = []

    # required columns for representations
    # added 'shape' as opposed to the original
    # so you have to modify __find_bulk_embeddings
    df_cols = [
        "identity",
        "hash",
        "embedding",
        "original_shape",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
    ]

    # Ensure the proper pickle file exists
    if not os.path.exists(datastore_path):
        with open(datastore_path, "wb") as f:
            pickle.dump([], f)

    # Load the representations from the pickle file
    with open(datastore_path, "rb") as f:
        representations = pickle.load(f)

    # check each item of representations list has required keys
    for i, current_representation in enumerate(representations):
        missing_keys = list(set(df_cols) - set(current_representation.keys()))
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )

    # embedded images
    pickled_images = [representation["identity"] for representation in representations]

    # Get the list of images on storage
    storage_images = image_utils.list_images(path=db_path)

    if len(storage_images) == 0:
        raise ValueError(f"No item found in {db_path}")

    # Enforce data consistency amongst on disk images and pickle file
    must_save_pickle = False
    new_images = list(set(storage_images) - set(pickled_images))  # images added to storage
    old_images = list(set(pickled_images) - set(storage_images))  # images removed from storage

    # detect replaced images
    replaced_images = []
    for current_representation in representations:
        identity = current_representation["identity"]
        if identity in old_images:
            continue
        alpha_hash = current_representation["hash"]
        # beta_hash = package_utils.find_hash_of_file(identity)
        beta_hash = image_utils.find_image_hash(identity)
        if alpha_hash != beta_hash:
            logger.debug(f"Even though {identity} represented before, it's replaced later.")
            replaced_images.append(identity)

    if not silent and (len(new_images) > 0 or len(old_images) > 0 or len(replaced_images) > 0):
        logger.info(
            f"Found {len(new_images)} newly added image(s)"
            f", {len(old_images)} removed image(s)"
            f", {len(replaced_images)} replaced image(s)."
        )

    # append replaced images into both old and new images. these will be dropped and re-added.
    new_images = new_images + replaced_images
    old_images = old_images + replaced_images

    # remove old images first
    if len(old_images) > 0:
        representations = [rep for rep in representations if rep["identity"] not in old_images]
        must_save_pickle = True

    # find representations for new images
    if len(new_images) > 0:
        representations += __find_bulk_embeddings(
            employees=new_images,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            normalization=normalization,
            silent=silent,
        )  # add new images
        must_save_pickle = True

    if must_save_pickle:
        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f)
        if not silent:
            logger.info(f"There are now {len(representations)} representations in {file_name}")

    # Should we have no representations bailout
    if len(representations) == 0:
        if not silent:
            toc = time.time()
            logger.info(f"find function duration {toc - tic} seconds")
        return []
    
    if not silent:
        toc = time.time()
        logger.info("Initialization complete")
        logger.info(f"find function duration {toc - tic} seconds")

    # post-processing
    representations = __find_missing_value(representations)
    representations = sorted(representations, key=operator.itemgetter('identity'))
    return representations


def __find_missing_value(
        representations: List[Dict["str", Any]]
    ) -> List[Dict["str", Any]]:
    """
    make missing value when target size is less then 2% of full size

    Args:
        representations (list): it contains dict that defined as df_col
    Returns:
        representations (list): processed List[Dict["str", Any]]
    """
    # modify percentage if you want to filter big target_size
    percentage = 2

    new_reps = []
    for rep in representations:
        original_size = rep['original_shape'][0] * rep['original_shape'][1]
        target_size = rep['target_w'] * rep['target_h']
        if original_size * percentage // 100 > target_size:
            rep['embedding'] = None
        new_reps.append(rep)
    return new_reps


# this function is copy from deepface.modules.recognition.__find_bulk_embeddings
def __find_bulk_embeddings(
        employees: List[str],
        model_name: str = MODEL_NAME,
        detector_backend: str = DETECTOR_BACKEND,
        enforce_detection: bool = ENFORCE_DETECTION,
        align: bool = ALIGN,
        expand_percentage: int = 0,
        normalization: str = "base",
        silent: bool = False,
    ) -> List[Dict["str", Any]]:
    """
    Find embeddings of a list of images

    Args:
        employees (list): list of exact image paths

        model_name (str): Model for face recognition. Options: VGG-Face, Facenet, Facenet512,
            OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet (default is VGG-Face).

        detector_backend (str): face detector model name

        enforce_detection (bool): set this to False if you
            want to proceed when you cannot detect any face

        align (bool): enable or disable alignment of image
            before feeding to facial recognition model

        expand_percentage (int): expand detected facial area with a
            percentage (default is 0).

        normalization (bool): normalization technique

        silent (bool): enable or disable informative logging
    Returns:
        representations (list): pivot list of dict with
            image name, hash, embedding and detected face area's coordinates
    """
    representations = []
    for employee in tqdm(
        employees,
        desc="Finding representations",
        disable=silent,
    ):
        file_hash = image_utils.find_image_hash(employee)

        try:
            img_objs = detection.extract_faces(
                img_path=employee,
                detector_backend=detector_backend,
                grayscale=False,
                enforce_detection=enforce_detection,
                align=align,
                expand_percentage=expand_percentage,
            )

        except ValueError as err:
            logger.error(f"Exception while extracting faces from {employee}: {str(err)}")
            img_objs = []

        if len(img_objs) == 0:
            representations.append(
                {
                    "identity": employee,
                    "hash": file_hash,
                    "embedding": None,
                    "original_shape": None,
                    "target_x": 0,
                    "target_y": 0,
                    "target_w": 0,
                    "target_h": 0,
                }
            )
        else:
            for img_obj in img_objs:
                img_content = img_obj["face"]
                img_region = img_obj["facial_area"]
                embedding_obj = representation.represent(
                    img_path=img_content,
                    model_name=model_name,
                    enforce_detection=enforce_detection,
                    detector_backend="skip",
                    align=align,
                    normalization=normalization,
                )

                img_representation = embedding_obj[0]["embedding"]
                representations.append(
                    {
                        "identity": employee,
                        "hash": file_hash,
                        "embedding": img_representation,
                        "original_shape": cv2.imread(employee).shape,
                        "target_x": img_region["x"],
                        "target_y": img_region["y"],
                        "target_w": img_region["w"],
                        "target_h": img_region["h"],
                    }
                )

    return representations
