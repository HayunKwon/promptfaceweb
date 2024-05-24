# promptface
customized deepface(v0.0.91) fork repo

## Installation
Recommend install promptface and deepface(v0.0.91) is to download it from github. It's going to install the library itself and its prerequisites as well.
```console
$ pip install git+https://github.com/M1nu0x0/promptface.git
```

Then you will be able to import the library and use its functionallities.
```python
from deepface import Deepface   # However the version of deepface is v0.0.91
from promptface.Promptface import Promptface
from promptface.utils.abstract import AbstractOnVeried
```

## How to use
### deepface
You can read README.md from [deepface](https://github.com/serengil/deepface) repository.

### promptface
You can use the promptface by running quick_start.py.

#### Quick Start Local .py
```py
# recommend python 3.8.12 version
# project dependencies
from promptface.utils.logger import Logger
from promptface.Promptface import Promptface
from promptface.utils.abstract import AbstractOnVeried

logger = Logger(__name__)


class MyCallback(AbstractOnVeried):
    def on_verify_success(self, app_instance: Promptface, *args, **kwargs):
        # path=str, distance=float, face_coordinate=[(x,y,w,h)]
        logger.info(f'{app_instance.target_path} {app_instance.target_distance} {app_instance.faces_coordinates}')
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
```

#### Constants.json
```json
{
    "DB_PATH": "./ImgDataBase",
    "MODEL_NAME": null,
    "DETECTOR_BACKEND": null,
    "ENFORCE_DETECTION": true,
    "ALIGN": true,
    "DISCARD_PERCENTAGE": 2,
    "SOURCE": 0,
    "TIME_THRESHOLD": 5,
    "FRAME_THRESHOLD": 10,
    "BROKER_IP": null,
    "BROKER_PORT": 1883,
    "TOPIC_STREAM": "home/stream",
    "TOPIC_RESULT": "home/result"
}
```
If you assign a `null` values in constants.json, it is automatically set to the values in the following table. But if you want to use `MQTT`, please write BROKER_IP.
| Constants          |    Contents     |
| ------------------ | :-------------: |
| DB_PATH            | "./ImgDataBase" |
| MODEL_NAME         |    "ArcFace"    |
| DETECTOR_BACKEND   |    "opencv"     |
| ENFORCE_DETECTION  |      True       |
| ALIGN              |      True       |
| DISCARD_PERCENTAGE |        2        |
| SOURCE             |        0        |
| TIME_THRESHOLD     |        5        |
| FRAME_THRESHOLD    |       10        |
| BROKER_IP          |      null       |
| TOPIC_STREAM       |  "home/stream"  |

There is options for model and detector.
```py
detectors = ["retinaface", "mtcnn", "fastmtcnn", "dlib", "yolov8", "yunet", "centerface", "mediapipe", "ssd", "opencv", "skip"]

models = ["Facenet512", "Facenet", "VGG-Face", "ArcFace", "Dlib", "GhostFaceNet", "SFace", "OpenFace", "DeepFace", "DeepID"]
```
If you want to use some models and detectors, requirements_additional.txt must be installed.

#### Folder Format
There is an example of a database folder format that was modified a little from Deepface Streaming.
```
root
├──main.py
├──constants.json
├──Logs
│   ├── promptface.log
├── ImgDataBase
│   ├── Alice
│   │   ├── Alice01.jpg
│   │   ├── Alice02.jpg
│   ├── Bob
│   │   ├── Bob01.jpg
```

Some logic has been changed in comparison to deepface
- The cosine_distance measurement method has been changed to use scikit-learn.
- Sort df in identity order.
- You can do something when verifies

#### MQTT
If you want to use MQTT protocol to use raspberrypi, etc as camera, use quick_raspi, quick_server instead of quick_start.

#### Quick Start RaspberryPI
```python
# project dependencies
from promptface.PublishCV2 import client


if __name__ == '__main__':
    client()
```

#### Quick Start Server
```python
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
        payload: PayloadType = f'{app_instance.target_path} {app_instance.target_distance} {app_instance.faces_coordinates}'
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
```
There is an payload which contains results about verify. This payload is an example, so you can do anything you want.

**I recommand that you type your own server ip in broker_ip.**

## Diffrence from deepface
### face size threshold
Default face size threshold in streaming is `130`. So I thought that threshold doesn't fit every resolutions like smaller size resolution and bigger size resolution. Here is an example resolution.

height x width (View camera shots long vertically)
- 1920 x 1080
  - I think FHD size resolution is optimized for default threshold, 130.
  - When it is MFS(Medium Full Shot, 3/4 shot), the face area is about 2%.
- 640 x 480
  - When this resolution, 2% is about 78 x 78.
  - So 130 is too BIG.
- 3840 x 2160
  - When this resolution, 2% is about 407 x 407.
  - So 130 is too SMALL.

Threshold formula.
```math
 t = \sqrt{\frac{h\times w\times x}{100}}
```
when h = height, w = width, x = ratio, t = threshold.

Here are areas of face I experimented with.
| Shot             | Percent    |
| ---------------- | ---------- |
| Medium Closed Up | about 3.6% |
| Medium Shot      | about 2.7% |
| Medium Full Shot | about 1.9% |

Caution. The ratio varies depending on the composition of your screen, so calculate the face area yourself and use it.

To calculate the ratio, use the following equation.
```math
x = \frac{h \times w}{H \times W}
```
when h = face height, w = face width, H = img height, W = img width.

### promptface.modules.pkl
This Module contains "show_pkl", "load_pkl", "init_pkl" funcs. Among them, "init_pkl" is a customized function from deepface.modules.recognitions: find. "find" function has two main logics. First, initialize pickle file. Second, find the minimum distance between the input img and DataFrame from pkl. So I made "init_pkl" because I wanted to separate the two features.

### promptface.modules.streaming
This Module is customized func from deepface.modules.streaming: stream. I want to do someting when "freezed". And I don't need other analysis tool. So I removed everyting except the classification. Finally, I felt inefficient to check pickle files for every face detection.

### Cosine Distance
As a result of separating the functions of "find" into two, we had to find the most similar image. At that time, what caught my eye was the benchmark of deepface(v0.0.91), and using cosine_distance worked well, so I implemented it through scikit-learn.

## License

PromptFace is licensed under the MIT License - see [`LICENSE`](https://github.com/M1nu0x0/prompt_face/blob/master/LICENSE) for more details.

## About DeepFace License
DeepFace wraps some external face recognition models: [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/), [Facenet](https://github.com/davidsandberg/facenet/blob/master/LICENSE.md), [OpenFace](https://github.com/iwantooxxoox/Keras-OpenFace/blob/master/LICENSE), [DeepFace](https://github.com/swghosh/DeepFace), [DeepID](https://github.com/Ruoyiran/DeepID/blob/master/LICENSE.md), [ArcFace](https://github.com/leondgarse/Keras_insightface/blob/master/LICENSE), [Dlib](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt), [SFace](https://github.com/opencv/opencv_zoo/blob/master/models/face_recognition_sface/LICENSE) and [GhostFaceNet](https://github.com/HamadYA/GhostFaceNets/blob/main/LICENSE). Besides, age, gender and race / ethnicity models were trained on the backbone of VGG-Face with transfer learning. Similarly, DeepFace wraps many face detectors: [OpenCv](https://github.com/opencv/opencv/blob/4.x/LICENSE), [Ssd](https://github.com/opencv/opencv/blob/master/LICENSE), [Dlib](https://github.com/davisking/dlib/blob/master/LICENSE.txt), [MtCnn](https://github.com/ipazc/mtcnn/blob/master/LICENSE), [Fast MtCnn](https://github.com/timesler/facenet-pytorch/blob/master/LICENSE.md), [RetinaFace](https://github.com/serengil/retinaface/blob/master/LICENSE), [MediaPipe](https://github.com/google/mediapipe/blob/master/LICENSE), [YuNet](https://github.com/ShiqiYu/libfacedetection/blob/master/LICENSE), [Yolo](https://github.com/derronqi/yolov8-face/blob/main/LICENSE) and [CenterFace](https://github.com/Star-Clouds/CenterFace/blob/master/LICENSE). License types will be inherited when you intend to utilize those models. Please check the license types of those models for production purposes.


DeepFace [logo](https://thenounproject.com/term/face-recognition/2965879/) is created by [Adrien Coquet](https://thenounproject.com/coquet_adrien/) and it is licensed under [Creative Commons: By Attribution 3.0 License](https://creativecommons.org/licenses/by/3.0/).
