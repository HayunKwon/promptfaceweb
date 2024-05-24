# 3rd-party dependencies
from paho.mqtt.client import PayloadType

# project dependencies
from promptface.utils.logger import Logger
from promptface.Subscriber import Subscriber
from promptface.utils.abstract import AbstractOnVeried

logger = Logger(__name__)

TOPIC_ON_VERIFY = 'home/onverify'


class MyCallback(AbstractOnVeried):
    def on_verify_success(self, app_instance: Subscriber, *args, **kwargs):
        # path=str, distance=float, face_coordinate=[(x,y,w,h)]
        # payload: PayloadType = f'{app_instance.target_path} {app_instance.target_distance} {app_instance.faces_coordinates}'
        # target label
        label = app_instance.target_path.split('\\')[-1].split('.')[0] if app_instance.target_path else None
        payload: PayloadType = str(label)
        app_instance.client.publish(TOPIC_ON_VERIFY, payload)
        logger.info(f'payload: {payload}')
        logger.info(f'args: {args}')
        logger.info(f'kwargs: {kwargs}')

    def on_verify_failed(self, app_instance: Subscriber, *args, **kwargs):
        # path=None, distance=float, face_coordinate=[(0,0,0,0)]
        payload: PayloadType = f'{app_instance.target_path} {app_instance.target_distance} {app_instance.faces_coordinates}'
        app_instance.client.publish(TOPIC_ON_VERIFY, payload)

        logger.info(f'payload: {payload}')
        logger.info(f'args: {args}')
        logger.info(f'kwargs: {kwargs}')


# Main
callback = MyCallback()
Subscriber.app(callback, 'this is args1', 2, key1=(), key2=('value1', 'value2'))