#!/usr/bin/env python
import sys
import os
import cv2
import pickle
import numpy as np
from matplotlib.image import imread

data_dir = "/media/miro/WD/jetbot_obstacle_avoidance"

size = 224

if __name__ == "__main__":
    if len(sys.argv) == 2:
        data_dir = sys.argv[1]
        if not os.path.isdir(data_dir):
            sys.exit(data_dir + " is not a directory")

    if len(sys.argv) == 3:
        try:
            size = int(sys.argv[2])
        except ValueError:
            sys.exit(sys.argv[2] + " is not integer value")

    if not os.path.exists(data_dir):
        sys.exit(data_dir + " does not exist")

    name_to_idx = dict()
    idx_to_name = dict()
    class_count = 0

    X = []
    y = []

    for c in os.listdir(data_dir):
        c_dir = os.path.join(data_dir, c)
        if os.path.isdir(c_dir):
            name_to_idx[c] = class_count
            idx_to_name[class_count] = c
            for f in os.listdir(c_dir):
                if os.path.splitext(f)[1] == ".jpg":
                    print(os.path.join(data_dir, c, f))
                    im = imread(os.path.join(data_dir, c, f))
                    # 2 ways to resize image: opencv and scikit-image
                    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
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
    with open(os.path.join(data_dir, 'data_' + str(size) + '.pckl'), 'wb') as f:
        pickle.dump([X, y, name_to_idx, idx_to_name], f)
        print("Exported to ", f.name)
