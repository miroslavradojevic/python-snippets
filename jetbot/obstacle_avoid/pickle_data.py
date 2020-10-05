#!/usr/bin/env python
import argparse
import sys
import os
import cv2
import pickle
import numpy as np
from matplotlib.image import imread
from os.path import isdir, join, exists, splitext
from os import listdir
from pathlib import Path


# -d /media/miro/WD/jetbot_obstacle_avoidance/data
# -d /media/miro/WD/L-CAS/LCAS_1200/data
# =size 224

if __name__ == "__main__":
    parser = argparse.ArgumentParser(help="")
    parser.add_argument("-d", type=str, required=True, help="Data directory")
    parser.add_argument("-size", type=int, default=0, help="Size of the output, no resize for size set to 0")
    parser.add_argument("-prefix", type=str, default="data", help="Pickle file prefix")
    args = parser.parse_args()

    if not exists(args.d):
        exit("{} does not exist".format(args.d))

    if not isdir(args.d):
        exit("{} is not a directory".format(args.d))

    name_to_idx = dict()
    idx_to_name = dict()
    class_count = 0

    X = []
    y = []

    for c in listdir(args.d):
        c_dir = join(args.d, c)
        if isdir(c_dir):
            name_to_idx[c] = class_count
            idx_to_name[class_count] = c
            for f in listdir(c_dir):
                if splitext(f)[-1] == ".jpg":
                    print(join(args.d, c, f))
                    im = imread(join(args.d, c, f))
                    # print("{} {} {} {} {}".format(type(im), im.shape, im[0].dtype, np.amin(im), np.amax(im)))
                    # im: <class 'numpy.ndarray'> (2464, 3280, 3) uint8 0 255

                    # 2 ways to resize image: opencv and scikit-image
                    if args.size != 0:
                        im = cv2.resize(im, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
                        # im: <class 'numpy.ndarray'> (224, 224, 3) uint8 0 255

                    X.append(im)
                    y.append(class_count)

            class_count = class_count + 1

    X = np.array(X)
    y = np.array(y).astype(np.uint16)
    print(name_to_idx)
    print(idx_to_name)
    print("X", X[0].dtype, X.shape)
    print("y", y[0].dtype, y.shape)

    # pickle data
    with open(join(Path(args.d).parent, args.prefix + "_" + str(args.size) + '.pckl'), 'wb') as f:
        pickle.dump([X, y, name_to_idx, idx_to_name], f)
        print("Exported to ", f.name)
