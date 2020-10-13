#!/usr/bin/env python
import argparse
import os
import time
from datetime import datetime
from os.path import expanduser, join, basename
import sys

from rpiv2 import Camera

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record video .avi using Jetson Nano with RPi_v2 camera using OpenCV")
    parser.add_argument("-fps", type=int, help="Frames per second - camera capture", default=21)
    parser.add_argument("-scale", type=int, help="", default=4)
    args = parser.parse_args()

    if args.scale < 1 or args.scale > 10:
        exit("Scale is out of accepted range")

    # output directory used when saving frames individualy
    # dir_path = join(expanduser("~"), "record_" + time.strftime("%Y%m%d-%H%M%S")) # basename(__file__)
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)

    video_record = True
    cam = Camera(Camera.RPI_V2_WIDTH // args.scale, Camera.RPI_V2_HEIGHT // args.scale, args.fps, video_record)

    # t0 = None
    # t1 = None

    try:
        while True:
            # cam.value would access frame captured by the camera
            # img_path = os.path.join(dir_path, 'cam_' + str(int(time.time() * 1000.0)) + '.jpg')
            # cv2.imwrite(img_path, cam.value)
            sys.stdout.write("{} recording...\r".format(datetime.now()))
            sys.stdout.flush()

            # if t0 is None:
            #     t0 = int(time.time() * 1000.0)
            # else:
            #     t1 = int(time.time() * 1000.0)
            #
            #
            #
            #     dt = t1 - t0  # milliseconds
            #     t0 = t1
            #     if dt > 0:
            #         print("FPS={:.5f}".format(1.0 / (dt / 1000.0)))  # img_path
            #     else:
            #         print("dt={:.5f}".format(dt))

            time.sleep(0.1)
    except KeyboardInterrupt:
        cam.stop()
        # cv2.destroyAllWindows()
