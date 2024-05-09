"""
This module is a modified version of deepface.modules.streaming.analysis function
"""

# 3rd party dependencies
import cv2  # for imshow, waitKey, destroyAllWindows

# deepface
from deepface import DeepFace   # for build_model

# project dependencies
from promptface.utils.constants import MODEL_NAME, INFO_FORMAT
from promptface.utils.logger import Logger
from promptface.utils.folder_utils import createDirectory
from promptface.modules.pkl import load_pkl
from promptface.modules.streaming import get_camera, process


# ----- INIT -----
try:
    # set logger
    createDirectory('Logs')
    logger = Logger(__name__)
    logger.info(INFO_FORMAT.format('APP START'))

    # build model
    _ = DeepFace.build_model(model_name=MODEL_NAME)
    logger.info(f"{MODEL_NAME} is built")

    # load pickle
    # database_embeddings are vetorized from embeddings col
    database_embeddings, identities = load_pkl()

    # activate camera
    cap = get_camera()  # webcam
except Exception as e:
    logger.critical(str(e))
    exit(1)


# ----- MAIN -----
logger.info(INFO_FORMAT.format('START MAIN'))
while True:
    # get img from new frame
    _, img = cap.read()

    # get verified content and img processing like boxing face, etc...
    img, target_path, target_distance = process(img, database_embeddings, identities)

    # show
    cv2.imshow("img", img)


    # enter only once when freeze
    # success
    if target_distance and target_path:
        target_label = target_path.split('/')[-1].split('\\')[-2]
        logger.info(f"Hello, {target_label} {target_distance} {target_path}")
        # --- do something like on/off green LEDs or save data, etc... ---

    # failure
    if target_distance and target_path is None:
        logger.info(f"Verify failed, {target_distance}")
        # --- do something like on/off red LEDs or save data, etc... ---

    process.target_distance = None  # this means enter success and failure once per freeze


    # user input
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
    try:
        key_code = cv2.waitKey(1) & 0xFF

        if key_code == ord("q"):
            logger.info(INFO_FORMAT.format('QUIT'))
            break
        if key_code == ord("r"):
            logger.info(INFO_FORMAT.format('RE-LOAD IMG-DATABASE'))
            database_embeddings, identities = load_pkl()
            logger.info(INFO_FORMAT.format('RE-START MAIN'))
    except Exception as e:
        logger.critical(str(e))
        break


# ----- END -----
# kill open cv things
cap.release()
cv2.destroyAllWindows()
logger.info(INFO_FORMAT.format('END'))
