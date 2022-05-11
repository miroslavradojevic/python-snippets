#!/usr/bin/env python3
# use all sorts of libraries available in python to read .tif image stack (3D) into numpy array
import argparse
import cv2
from os import exists
# from matplotlib.image import imread
import skimage
# print(skimage.__version__)
from skimage import io

im_path = 'screen-test-rgb.jpg'

# pip install ?
im_cv = cv2.imread(im_path)
im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
print(type(im_cv), im_cv.shape)

# 
im_matplotlib = imread(im_path)
print(type(im_matplotlib), im_matplotlib.shape)


# https://scikit-image.org/docs/stable/install.html
# pip install scikit-image
im_skimage = io.imread(im_path)

print(type(im_skimage), im_skimage.shape)

# if not os.path.isfile(image_path):
#     print(image_path, " is not a file.")
#     quit()

# if not image_path.endswith(".tif"):
#     print(image_path, " needs to be tif (stack).")

# imageio
# from scipy.misc import imread

if __name__=='__main__':
    psr = argparse.ArgumentParser(description='Read .tif image stack')
    psr.add_argument('--f', type=str, required=True, help='Path to the .tif file (image stack)')

    args = psr.parse_args()

    if not exists(args.f):
        print()
        exit(1)