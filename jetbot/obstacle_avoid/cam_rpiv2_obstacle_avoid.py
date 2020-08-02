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
import numpy as np
sys.path.insert(1, join(sys.path[0], dirname(sys.path[0])))
from cam_rpiv2 import CameraRPiv2
from jetbot import Robot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read camera image and classify whether it is obstacle. Use RPiv2 capture to feed classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model", help="Path to h5 model", type=str)

    args = parser.parse_args()

    if not exists(args.model) or not isfile(args.model) or not splitext(args.model)[1] == ".h5":
        exit(args.model + "error: does not exist, is not a file or needs to have .h5 extension")

    model = load_model(args.model)
    input_shape = model.layers[0].input_shape
    # print(input_shape, input_shape[1:3], type(input_shape[1:3]), flush=True)

    cam = CameraRPiv2()
    print("Camera initialized. Capturing...", flush=True)

    robot = Robot()
    print("Robot initialized...", flush=True)
    # period_mean = None
    # period_count = 0
    try:
        while True:
            im = cv2.resize(cam.value, input_shape[1:3], interpolation=cv2.INTER_CUBIC)
            im = np.expand_dims(im, axis=0)
            im = im / 255.0

            t1 = time.time()
            preds =  model.predict(im)
            t2 = time.time()

            # if period_mean is None:
            #     period_mean = t2-t1
            #     period_count = 1
            # else:
            #     period_count += 1
            #     period_mean = period_mean * ((period_count-1) / period_count) + (t2-t1) / period_count

            print(1./(t2 - t1), ",")

            if np.argmax(preds, axis=1) == 0:
                robot.set_motors(0.35, 0.35)
            else:
                robot.stop()

    except KeyboardInterrupt:
        cam.stop()
