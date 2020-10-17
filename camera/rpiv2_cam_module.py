import traitlets
from traitlets.config.configurable import SingletonConfigurable
import atexit
import cv2
import time
import os
from os.path import expanduser
import sys
import threading
import numpy as np


class Camera(SingletonConfigurable):
    value = traitlets.Any()

    # config
    fps = traitlets.Integer(default_value=21).tag(config=True)
    capture_width = traitlets.Integer(default_value=3280).tag(config=True)
    capture_height = traitlets.Integer(default_value=2464).tag(config=True)
    width = traitlets.Integer(default_value=3280).tag(config=True)  # // 4 224 resize for 224x224 neural net
    height = traitlets.Integer(default_value=2464).tag(config=True)  # // 4 224

    def __init__(self, *args, **kwargs):
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        super(Camera, self).__init__(*args, **kwargs)

        try:
            print(self._gst_str())
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            re, image = self.cap.read()
            if not re:
                raise RuntimeError('Could not read image from camera.')

            self.value = image
            self.start()
        except:
            self.stop()
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        while True:
            re, image = self.cap.read()
            if re:
                self.value = image
            else:
                break

    def _gst_str(self):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)RGBx ! videoconvert ! appsink' % (
            self.capture_width, self.capture_height, self.fps, self.width, self.height)

    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()

    def restart(self):
        self.stop()
        self.start()


if __name__ == '__main__':
    # fps = 24  # TODO add as argument
    dir_path = os.path.join(expanduser("~"), os.path.basename(__file__) + "_" + time.strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cam = Camera()

    print("\n\nExporting to ", dir_path)

    t0 = None
    t1 = None

    try:
        while True:
            if t0 is None:
                t0 = int(time.time() * 1000.0)
            else:
                t1 = int(time.time() * 1000.0)
                img_path = os.path.join(dir_path, 'cam_' + str(t1) + '.jpg')
                cv2.imwrite(img_path, cam.value)
                dt = t1 - t0  # milliseconds
                t0 = t1
                if dt>0:
                    print("FPS={:.2f}".format(1.0 / (dt / 1000.0)))  # img_path

            # time.sleep(1e-6)
    except KeyboardInterrupt:
        cam.stop()
