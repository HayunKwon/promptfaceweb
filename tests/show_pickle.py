"""
This module is for pickle to see what is in it
"""

# built-in dependencies
import copy

# 3rd party dependencies
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# project dependencies
from promptface.modules.pkl import init_pkl
from promptface.utils.logger import Logger
from promptface.utils.constants import INFO_FORMAT

# set is_window = True when you want to see faces in db
is_window = False


# ----- INIT -----
try:
    # set logger
    logger = Logger(__name__, 'Logs/show_pickel.log')
    logger.info(INFO_FORMAT.format('APP START'))

    # init pickle and get representatinos
    representations = init_pkl()
except Exception as e:
    logger.critical(str(e))
    exit(1)


# ----- MAIN -----
logger.info(INFO_FORMAT.format('SHOW PICKLE'))
df = pd.DataFrame(representations)
logger.info('\n{}'.format(df))

# img_paths = representations.identity
for _, data_row in df.iterrows():
    # set data
    path = data_row.identity
    is_face = True if data_row.embedding else False
    logger.info('{}\tface area ratio: {}%'.format(path, round((data_row.target_w * data_row.target_h) / (data_row.original_shape[0] * data_row.original_shape[1]) * 100, 2)))

    if is_window is False:
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
