#!/usr/bin/env python
import traitlets
import threading
import atexit
import cv2
import time
import numpy as np
from traitlets.config.configurable import SingletonConfigurable

class Camera(SingletonConfigurable):
    # fixed resolution of the Raspberry Pi v2 camera
    RPI_V2_WIDTH = 3280
    RPI_V2_HEIGHT = 2464
    FPS_MAX = 21
    value = traitlets.Any()
    # config
    fps = traitlets.Integer(default_value=FPS_MAX).tag(config=True)
    capture_width = traitlets.Integer(default_value=RPI_V2_WIDTH).tag(config=True)
    capture_height = traitlets.Integer(default_value=RPI_V2_HEIGHT).tag(config=True)
    width = traitlets.Integer(default_value=RPI_V2_WIDTH).tag(config=True)  # 224 resize for 224x224 neural net
    height = traitlets.Integer(default_value=RPI_V2_HEIGHT).tag(config=True)  # 224

    def _gst_str(self):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
            self.capture_width, self.capture_height, self.fps, self.width, self.height)

    def __init__(self, cap_width, cap_height, cap_fps, write_video, *args, **kwargs):
        self.width = cap_width
        self.height = cap_height

        if cap_fps > self.FPS_MAX or cap_fps < 1:
            raise RuntimeError('Could not set FPS={}'.format(cap_fps))
        self.fps = cap_fps

        self.write_video = write_video
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)

        super(Camera, self).__init__(*args, **kwargs)

        try:
            print(self._gst_str())
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            
            if self.write_video:
                # https://www.geeksforgeeks.org/saving-a-video-using-opencv/
                cap_width = int(self.cap.get(3))
                cap_height = int(self.cap.get(4))
                self.result = cv2.VideoWriter("record_" + time.strftime("%Y%m%d-%H%M%S") + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (cap_width, cap_height))

            re, image = self.cap.read()

            if re:
                self.value = image
                if self.write_video:
                    self.result.write(image)
            else:
                raise RuntimeError('Could not read image from camera.')

            self.start()
        except:
            self.stop()
            raise RuntimeError('Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        while True:
            re, image = self.cap.read()
            if re:
                self.value = image
                if self.write_video:
                    self.result.write(image)
            else:
                break

    def start(self):
        if not self.cap.isOpened():
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'result'):
            self.result.release()
        if hasattr(self, 'thread'):
            self.thread.join()

    def restart(self):
        self.stop()
        self.start()

def testfun():
    return "This is testfun()"