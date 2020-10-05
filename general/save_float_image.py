#!/usr/bin/env python
import numpy as np
import cv2

if __name__ == '__main__':
    arr = np.random.rand(512, 512) * 255
    print("arr {} | {} -- {} | {}".format(arr.shape, np.amin(arr), np.amax(arr), arr.dtype))
    cv2.imwrite("arr.jpg", arr)
