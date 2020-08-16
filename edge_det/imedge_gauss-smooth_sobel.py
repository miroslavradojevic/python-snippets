#!/usr/bin/env python
import argparse
import numpy as np
import cv2

from matplotlib.image import imread


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute image edges" + \
                    " using Gaussian smoothing and Sobel edge detector.")
    parser.add_argument("img", help="Path to image", type=str)
    # parser.add_argument()
    args = parser.parse_args()





