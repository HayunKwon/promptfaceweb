# project dependencies
from promptface.utils.logger import Logger
from promptface.Promptface import Promptface
from promptface.utils.abstract import AbstractOnVeried

logger = Logger(__name__)


class MyCallback(AbstractOnVeried):
    def on_verify_success(self, app_instance: Promptface, *args, **kwargs):
        # path=str, distance=float, face_coordinate=[(x,y,w,h)]
        # logger.info(f'{app_instance.target_path} {app_instance.target_distance} {app_instance.faces_coordinates}')
        logger.info(str(app_instance.target_path.split('\\')[-2]))
        logger.info(f'args: {args}')
        logger.info(f'kwargs: {kwargs}')

    def on_verify_failed(self, app_instance: Promptface, *args, **kwargs):
        # path=None, distance=float, face_coordinate=[(0,0,0,0)]
        logger.info(f'{app_instance.target_path} {app_instance.target_distance} {app_instance.faces_coordinates}')
        logger.info(f'args: {args}')
        logger.info(f'kwargs: {kwargs}')


# Main
callback = MyCallback()
Promptface.app(callback, 'this is args1', 2, key1=(), key2=('value1', 'value2'))
