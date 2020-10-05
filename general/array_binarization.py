#!/usr/bin/env python
import numpy as np

if __name__ == '__main__':
    arr = np.random.rand(20, 4)
    print(arr.dtype)
    print(arr.shape)

    # binarize array
    arr.tofile("arr.bin")

    # read binary file
    arr1 = np.fromfile("arr.bin", dtype=arr.dtype)
    arr1 = arr1.reshape((-1, 4))
    print(arr1.shape)
    print(arr == arr1)

