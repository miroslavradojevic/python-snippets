#!/usr/bin/env python
# Image capture with RPi Camera Module v2, resolution 3280 x 2464 pixels
# Apply classification using trained model read from .h5 file
import argparse
import sys
import cv2
import time
from os.path import splitext, exists, isfile, join, dirname
from tensorflow.keras.models import load_model
import tensorflow as tf
sys.path.insert(1, join(sys.path[0], dirname(sys.path[0])))
from cam_rpiv2 import CameraRPiv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read camera image and classify whether it is obstacle. Use RPiv2 camera recording to feed classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", help="Path to h5 model", type=str)

    args = parser.parse_args()

    if not exists(args.model):
        exit(args.model + "does not exist")

    if not isfile(args.model):
        exit(args.model + "is not a file")

    if not splitext(args.model)[1] == ".h5":
        exit(args.model + "needs to have .h5 extension")

    model = load_model(args.model)
    print(model.layers[0].input_shape)
    print(type(model.layers[0].input_shape))

    cam = CameraRPiv2()

    try:
        while True:
            # img_path = join(d_out, 'frame_'+ datetime.now().strftime("%Y%m%d-%H%M%S-%f") +'.jpg')
            # logging.info("Capture camera value and write to {}".format(img_path))
            # cv2.imwrite(img_path, cam.value)
            # print(img_path)
            print(type(cam.value), cam.value.shape)
            time.sleep(0.001)
    except KeyboardInterrupt:
        cam.stop()
