import logging
import os
import sys
import cv2
import time
from datetime import datetime
import threading

# Test periodical image capture with
# RPi Camera Module v2, resolution 3280 x 2464 pixels

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class CameraRPiv2(): # cap_w=3280, cap_h=2464
    capture_width = 3280
    capture_height = 2464
    fps = 21
    width = 3280 # // 2
    height = 2464 # // 2
    
    def __init__(self, *args, **kwargs):
        # super(CameraRPiv2, self).__init__(*args, **kwargs)
        # self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)
        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            re, image = self.cap.read()
            if not re:
                raise RuntimeError('Could not read image from camera.')
            self.value = image
            self.start()
        except:
            self.stop()
            raise RuntimeError('Could not initialize camera')
        # atexit.register(self.stop)

    def _gst_str(self):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                self.capture_width, self.capture_height, self.fps, self.width, self.height)
        
    def _capture_frames(self):
        while True:
            re, image = self.cap.read()
            if re:
                self.value = image
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
        if hasattr(self, 'thread'):
            self.thread.join()
            
    def restart(self):
        self.stop()
        self.start()


if __name__ == "__main__":
    
    d_name = os.path.abspath(os.path.dirname(__file__))
    f_name = os.path.splitext(os.path.basename(__file__))[0]

    logging.basicConfig(
        level=logging.DEBUG, 
        filename=os.path.join(d_name, f_name + ".log"),
        filemode="w",
        format="%(asctime)s: %(levelname)s: %(funcName)s Line:%(lineno)d %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S")

    logging.info("Starting camera test")

    secs = 0.5
    if len(sys.argv) == 2 and is_number(sys.argv[1]): #  value.isdigit()
        secs = float(sys.argv[1])
        logging.info("Set period to {} second(s)".format(secs))
    else:
        print("Test whether camera is periodically capturing images")
        print("Usage: python3 " + __file__ + " nsec")
        sys.exit()

    d_out = os.path.join(d_name, f_name)
    
    if not os.path.exists(d_out):
        logging.info("Make output directory {}".format(d_out))
        os.makedirs(d_out)
    
    cam = CameraRPiv2()

    try:
        while True:
            # img_path = os.path.join(d_out_0, datetime.now().strftime("%Y%m%d-%H%M%S-%f") + '.jpg')
            # img_path = os.path.join(d_out, 'frame_'+ datetime.now().strftime("%Y%m%d-%H%M%S-%f") +'.jpg')
            img_path = os.path.join(d_out, datetime.now().strftime("%Y%m%d-%H%M%S-%f") +'.jpg')
            logging.info("Capture camera value and write to {}".format(img_path))
            cv2.imwrite(img_path, cam.value)
            print(img_path)
            # <class 'numpy.ndarray'> (2464, 3280, 3)
            time.sleep(secs)
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt")
        cam.stop()