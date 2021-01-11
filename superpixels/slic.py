#!/usr/bin/env python
# superpixels
# Extract super-pixels as a pre-processing stage for computer vision algorithms
import argparse
from os.path import exists, isfile, splitext
import numpy as np
from matplotlib.image import imread

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cluster 2D image into superpixels in order to take out sparser collection of positions with homogeneous intensity surrounding and avoid processing every pixel/voxel")
    parser.add_argument("-input", type=str, required=True,
                        help="Path to image file")
    # parser.add_argument("-cam", type=str, default=None, help="Path to directory with jpgs")
    args = parser.parse_args()

    ext = splitext(args.input)[-1].lower()
    if exists(args.input) and isfile(args.input) and ext in [".jpg"]:
        print(args.input)
        img = imread(args.input)
    else:
        print("Path {} does not exist".format())
