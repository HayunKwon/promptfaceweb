# built-in dependencies
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

'''
    DEBUG: 프로그램이 작동하는지 진단할 때 사용
    INFO: 프로그램이 예상대로 작동하는지 확인할 때 사용
    WARNING: 예상하지 못한 일이 발생했거나 가까운 미래에 발생할 문제가 있을 때 보고, (프로그램은 게속 동작)
    ERROR: 심각한 소프트웨어 문제로 일부 기능이 작동하지 못할 때, 예외를 일으키지 않으면서 에러 보고
    CRITICAL: 심각한 에러로 프로그램 자체가 계속 실행될 수 없을 때, 예외를 일으키지 않으면서 에러 보고
'''

class Logger:
    def __init__(self, module_name, file_name = 'Logs/promptface.log'):
        # init logger name, level
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel(logging.INFO)

        # set formatter
        _formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

        # set stream handler
        _stream_handler = logging.StreamHandler()
        _stream_handler.setFormatter(_formatter)

        # set file handler, it resets on monday
        try:
            if not os.path.exists('Logs'):
                os.makedirs('Logs')
        except OSError:
            print('Error: Failed to create directory: Logs')
        _file_handler = TimedRotatingFileHandler(filename = file_name, # filename
                                                 when="w0",             # monday
                                                 interval=1,            # every week
                                                 backupCount=52,        # 52 weeks = 1 year
                                                 encoding="utf-8")
        _file_handler.setFormatter(_formatter)

        # add handlers
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        self.logger.addHandler(_stream_handler)
        self.logger.addHandler(_file_handler)

    def set_level(self, level):
        self.logger.setLevel(level)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.logger.error(msg, *args, exc_info=exc_info, **kwargs)

# main
if __name__ == '__main__':
    my_logger = Logger('logger')
    my_logger.set_level(DEBUG)
    my_logger.debug('test debug')
    my_logger.info('test info')
    my_logger.warning('test warning')
    my_logger.error('test error')
    my_logger.critical('test critical')