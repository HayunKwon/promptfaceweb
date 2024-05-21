# built-in dependencies
import time

# 3rd-party dependencies
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# project dependencies
from promptface.utils.constants import INFO_FORMAT
from promptface.utils.logger import Logger
from promptface.modules.streaming import get_camera

logger = Logger(__name__)


class Publisher:
    def __init__(self, broker:str, port:int, topic:str):
        # IP Address of the Brocker
        self.broker_ip = broker
        self.port = port
        # ex) home/server
        self.topic = topic

    def client(self):
        # Object to capture the frames
        cap = get_camera()
        # Phao-MQTT Clinet
        client = mqtt.Client()
        # Establishing Connection with the Broker
        client.connect(self.broker_ip, self.port)
        try:
            while True:
                start = time.time()
                _, img = cap.read()

                # Encoding the Frame
                _, buffer = cv2.imencode('.jpg', img)
                # Converting into encoded bytes
                jpg_as_text = np.array(buffer).tobytes()

                # Publishig the Frame on the Topic home/server
                client.publish(self.topic, jpg_as_text)
                end = time.time()
                t = end - start
                fps = 1/t
                logger.info(fps)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info(INFO_FORMAT.format('QUIT'))
                    break
        except Exception as e:
            cap.release()
            client.disconnect()
            logger.info(str(e))