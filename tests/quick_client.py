# built-in dependencies
import time

# 3rd-party dependencies
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# project dependencies
from promptface.utils.logger import Logger
from promptface.utils.constants import RASPI_IP, RASPI_PORT, RASPI_TOPIC
from promptface.Publisher import client

logger = Logger(__name__)


if __name__ == '__main__':
    client(RASPI_IP, RASPI_PORT, RASPI_TOPIC)