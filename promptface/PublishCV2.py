# built-in dependencies
import base64

# 3rd-party dependencies
import cv2
import paho.mqtt.client as mqtt

# project dependencies
from promptface.utils.constants import SOURCE, BROKER_IP, BROKER_PORT, TOPIC_STREAM
from promptface.utils.logger import Logger

logger = Logger(__name__)


def on_connect(client:mqtt.Client, userdata, flags, rc):
    logger.info("Connected with result code " + str(rc))


def client(broker_ip=BROKER_IP, broker_port=BROKER_PORT, topic_stream=TOPIC_STREAM):
    # Object to capture the frames
    cap = cv2.VideoCapture(SOURCE)

    # Phao-MQTT Clinet
    client = mqtt.Client()
    client.on_connect = on_connect

    # Establishing Connection with the Broker
    client.connect(host=broker_ip, port=broker_port)
    client.loop_start()
    try:
        while True:
            # start = time.time()
            has_frame, img = cap.read()
            if has_frame is None:
                raise ValueError('No Camera')

            # Encoding the Frame
            _, buffer = cv2.imencode('.jpg', img)
            # Converting into encoded bytes
            jpg_as_text = base64.b64encode(buffer) # type: ignore

            # Publishig the img on the Topic home/stream
            client.publish(topic_stream, jpg_as_text)
            # end = time.time()
            # t = end - start
            # fps = 1/t
            # logger.info(fps)
    except Exception as e:
        cap.release()
        client.loop_stop()
        client.disconnect()
        logger.info(str(e))