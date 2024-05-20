# project dependencies
from promptface.MQTTPromptface import Subscriber
from promptface.utils.abstract import AbstractOnVeried

class MyCallback(AbstractOnVeried):
    def on_verify_success(self, app_instance: Subscriber, *args, **kwargs):
        target_path = app_instance.target_path
        target_distance = app_instance.target_distance
        print('{} {}'.format(target_path, target_distance))
        print(f'args: {args}')
        print(f'kwargs: {kwargs}')

    def on_verify_failed(self, app_instance: Subscriber, *args, **kwargs):
        target_path = app_instance.target_path
        target_distance = app_instance.target_distance
        print('{} {}'.format(target_path, target_distance))


# Main
callback = MyCallback()
Subscriber.app(callback, 'this is args1', 2, key1=(), key2=('value1', 'value2'))