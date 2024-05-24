# built-in dependencies
import os
import shutil

if __name__ == '__main__':
    import sys
    # add parant dir
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

# project dependencies
from promptface.utils.logger import Logger

logger = Logger(__name__)

def isDirectory(directory) -> bool:
    try:
        if os.path.exists(directory):
            logger.debug('directory exists: %s' % (directory))
            return True
        logger.debug('directory not exists: %s' % (directory))
        return False
    except OSError:
        logger.error('Failed to search the directory: %s' % (directory))
        return False

def createDirectory(directory) -> None:
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.debug('Success to create directory: %s' % (directory))
            return
    except OSError:
        logger.error('Failed to create the directory: %s' % (directory))
        return

def removeDirectory(directory, enforce=False) -> None:
    try:
        if os.path.exists(directory):
            if enforce == False:
                os.rmdir(directory)
            else:
                shutil.rmtree(directory)
            logger.debug('Success to remove directory: %s' % (directory))
            return
    except OSError:
        logger.error('Failed to remove the directory: %s' % (directory))
        return
    logger.warning('No directory: %s' % (directory))
    return

if __name__ == '__main__':
    from promptface.utils.logger import DEBUG
    logger.set_level(DEBUG)
    createDirectory('asdf')
    os.system('pause')
    removeDirectory('asdf')