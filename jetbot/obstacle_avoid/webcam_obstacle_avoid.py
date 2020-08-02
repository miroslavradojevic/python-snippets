#!/usr/bin/env python
import cv2
import argparse
import time
from tensorflow.keras.models import load_model
from os.path import splitext, exists, isfile, join, dirname
import numpy as np
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read webcam apply model.")
    parser.add_argument("model", help="Path to h5 model", type=str)

    args = parser.parse_args()

    if not exists(args.model) or not isfile(args.model) or not splitext(args.model)[1] == ".h5":
        exit("error" + args.model)

    model = load_model(args.model)
    input_shape = model.layers[0].input_shape

    # Connect to default web-camera
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("frame dimensions, w=%d, h=%d" % (width, height))

    try:
        while True:
            ret, frame = cap.read()
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, input_shape[1:3], interpolation=cv2.INTER_CUBIC)
            im = np.expand_dims(im, axis=0)
            im = im / 255.0

            t1 = time.time()
            preds = model.predict(im)
            t2 = time.time()
            print(1. / (t2 - t1), ",")
            # use preds fruther

    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()