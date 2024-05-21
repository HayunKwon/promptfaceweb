# built-in dependencies
import time

# 3rd-party dependencies
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# project dependencies
from promptface.utils.logger import Logger
from promptface.utils.constants import INFO_FORMAT, RASPI_IP, RASPI_PORT, RASPI_TOPIC

logger = Logger(__name__)


def client(broker, port, topic):
    # Object to capture the frames
    cap = cv2.VideoCapture(0)

    # Phao-MQTT Clinet
    client = mqtt.Client()
    # Establishing Connection with the Broker
    client.connect(broker, port)
    try:
        while True:
            start = time.time()
            has_frame, img = cap.read()
            if has_frame is None:
                raise ValueError('No Camera')

            # Encoding the Frame
            _, buffer = cv2.imencode('.jpg', img)
            # Converting into encoded bytes
            jpg_as_text = np.array(buffer).tobytes()

            # Publishig the Frame on the Topic home/server
            client.publish(topic, jpg_as_text)
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

if __name__ == '__main__':
    client(RASPI_IP, RASPI_PORT, RASPI_TOPIC)