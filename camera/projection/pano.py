#!/usr/bin/env python
# Project point-cloud on a spherical panorama image
# Input: point-cloud and spherical panorama
# Output: projection
# Adjusted from matlab functions: https://drive.google.com/file/d/1aeYfmquaivnUWWTjJ6kBT1jtPEihr-1g/view?usp=sharing
# Dataset: https://zenodo.org/record/2640062

from os.path import dirname, abspath, join
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
from matplotlib.image import imread

data_path = dirname(abspath(__file__))  # "/media/miro/WD/kucl_dataset/indoor"
pair_ind = 2
# [np.pi/2, np.pi/2, 0]

# Rigid-body transformation between a Ladybug5 and a LIDAR
R_LIDAR_to_LB = R.from_euler('ZYX',
                             np.deg2rad([-55.14, 15.47, -0.18])).as_dcm()  # Rotation: [yaw, pitch, roll] in degrees
print(np.round(R_LIDAR_to_LB, 3), type(R_LIDAR_to_LB), R_LIDAR_to_LB.shape)

t_LIDAR_to_LB = np.array([0.0934, 0.0597, -0.1659]).reshape(3, 1)  # Translation [tx, ty, tz] in meters
print(np.round(t_LIDAR_to_LB, 3), type(t_LIDAR_to_LB), t_LIDAR_to_LB.shape)

# Load image
img = imread(join(data_path, "images", "pano", "image{:04d}.png".format(pair_ind)))
# print(type(img), img.shape)

w = img.shape[1]
h = img.shape[0]


def xyz2pixel(pt_LB, w, h):
    # Range from Ladybug5 to points
    r = np.sqrt(np.multiply(pt_LB[0, :], pt_LB[0, :]) + \
                np.multiply(pt_LB[1, :], pt_LB[1, :]) + \
                np.multiply(pt_LB[2, :], pt_LB[2, :]))
    # print("r", type(r), r.shape)
    # uv of points
    th = np.arctan2(pt_LB[1, :], pt_LB[0, :])
    phi = np.arcsin(np.divide(pt_LB[2, :], r))
    # print("th", type(th), th.shape)
    u = -1 / (2 * math.pi) * th + 0.5
    v = -1 / math.pi * phi + 0.5
    # print("u", type(u), u.shape)
    # print("v", type(v), v.shape)
    # Pixels of points
    pixel = np.vstack((u * w, v * h))
    # print("pixel", type(pixel), pixel.shape)
    return pixel

def read_int(input_string):
    try:
        return int(input_string)
    except ValueError:
        return None


# Load points and reflectance
data = []
with open(join(data_path, "scans", "scan{:04d}.txt".format(pair_ind))) as f:
    n_lines = read_int(f.readline())
    if n_lines is not None:
        line = True
        while line:
            line = f.readline()
            line_vals = line.split(" ")
            if (len(line_vals) == 4):
                data.append([float(i) for i in line_vals])

data = np.array(data).transpose()
# print(type(data), data.shape)
pt_LIDAR = data[:3, :]
# reflectance = data[3, :]

# print(type(R_LIDAR_to_LB), R_LIDAR_to_LB.shape)
# print(type(pt_LIDAR), pt_LIDAR.shape)
# print(type(t_LIDAR_to_LB), t_LIDAR_to_LB.shape)

# Transform points w.r.t. LIDAR coordinate frame to Ladybug5 coordinate frame
pt_LB = np.matmul(R_LIDAR_to_LB, pt_LIDAR) + t_LIDAR_to_LB
# print(type(pt_LB), pt_LB.shape)

# Projection to spherical image
pix = xyz2pixel(pt_LB, w, h)
pix = np.round(pix, 0).astype(np.int) # round and cast as integer

# Visual verification
# img[pix[1, :], pix[0, :], :3] = 1, 0, 0
fig = plt.figure()
plt.imshow(img)
plt.scatter(x=pix[0, :], y=pix[1, :], marker=',',  c='r', lw=0, s=0.1)
# plt.show()
fig.tight_layout()
fig.savefig(join(data_path, "match{:04d}.png".format(pair_ind)), dpi=600, bbox_inches='tight')
# plt.close()




