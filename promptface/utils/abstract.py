from abc import ABC, abstractmethod

class AbstractOnVeried(ABC):
    @abstractmethod
    def on_verify_success(self, app_instance, *args, **kwargs):
        pass

    @abstractmethod
    def on_verify_failed(self, app_instance, *args, **kwargs):
        pass