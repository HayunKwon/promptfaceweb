"""
This module is a modified version of deepface.modules.streaming.analysis function

app: verify webcam from db
"""

# 3rd party dependencies
import cv2  # for imshow, waitKey, destroyAllWindows

# deepface
from deepface import DeepFace   # for build_model

# project dependencies
from promptface.utils.constants import DB_PATH, MODEL_NAME, INFO_FORMAT
from promptface.utils.logger import Logger
from promptface.utils.static import static_vars
from promptface.utils.folder_utils import createDirectory
from promptface.modules.pkl import load_pkl
from promptface.modules.streaming import get_camera, process

def cli() -> None:
    """
    command line interface function will be offered in this block
    """
    import fire
    fire.Fire()


@static_vars(
        target_path=None,
        target_distance=None
    )
def app(*args, **kwargs):
    """
    this function is from analysis.
    you can pass the (functions, params) to use

    Args:
        *args: functions
        **kwargs: params
    """
    # [0] is the (function, params) for when verify success
    # [1] is the (function, params) for when verify failed
    on_verify_success = args[0] if args[0] else None
    on_verify_failure = args[1] if args[1] else None
    params1 = kwargs.get("params1", ())
    params2 = kwargs.get("params2", ())

    # ----- INIT -----
    try:
        # set logger
        createDirectory(DB_PATH)
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
        img, app.target_path, app.target_distance = process(img, database_embeddings, identities)

        # show
        cv2.imshow("img", img)


        # enter only once when freeze
        # success
        if app.target_distance and app.target_path:
            target_label = app.target_path.split('/')[-1].split('\\')[-2]
            logger.info(f"Hello, {target_label} {app.target_distance} {app.target_path}")
            # --- do something like on/off green LEDs or save data, etc... ---
            if on_verify_success:
                on_verify_success(*params1)

        # failure
        if app.target_distance and app.target_path is None:
            logger.info(f"Verify failed, {app.target_distance}")
            # --- do something like on/off red LEDs or save data, etc... ---
            if on_verify_failure:
                on_verify_failure(*params2)

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
