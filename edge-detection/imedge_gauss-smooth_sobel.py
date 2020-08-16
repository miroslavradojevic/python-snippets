#!/usr/bin/env python
import argparse
import numpy as np
import cv2

from matplotlib.image import imread

def edge_detection(img, gauss_smooth, ):
    # Smooth image
    img = cv2.blur(img, (gauss_smooth, gauss_smooth))
    # Sobel
    # http://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute image edges" + \
                    " using Gaussian smoothing and Sobel edge detector.")
    parser.add_argument("img", help="Path to image", type=str)
    # parser.add_argument()
    args = parser.parse_args()

    # Load image
    img = imread(args.img)
    print(img.shape, type(img), img[0].dtype, np.min(img), np.max(img))

    # img = cv2.imread(args.img)
    # print(img.shape, type(img), img[0].dtype, np.min(img), np.max(img))

    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

