#!/usr/bin/env python
# Project point-cloud on a spherical panorama image
# Input: point-cloud and spherical panorama
# Output: pointcloud
# Adjusted from matlab functions: https://drive.google.com/file/d/1aeYfmquaivnUWWTjJ6kBT1jtPEihr-1g/view?usp=sharing
# Dataset: https://zenodo.org/record/2640062

from os.path import dirname, abspath, join
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
import argparse
from scipy.spatial.transform import Rotation as R
from matplotlib.image import imread


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


def project_pointcloud(im, ptcloud, rot, tr):
    # Rigid-body transformation between a Ladybug5 and a LIDAR
    R_LIDAR_to_LB = R.from_euler('ZYX', np.deg2rad(rot)).as_dcm()  # Rotation: [yaw, pitch, roll] in degrees
    print(np.round(R_LIDAR_to_LB, 3))

    t_LIDAR_to_LB = np.array(tr).reshape(3, 1)  # Translation [tx, ty, tz] in meters
    print(np.round(t_LIDAR_to_LB, 3))

    w = im.shape[1]
    h = im.shape[0]

    # print(type(R_LIDAR_to_LB), R_LIDAR_to_LB.shape)
    # print(type(pt_LIDAR), pt_LIDAR.shape)
    # print(type(t_LIDAR_to_LB), t_LIDAR_to_LB.shape)

    # Transform points w.r.t. LIDAR coordinate frame to Ladybug5 coordinate frame
    pt_LB = np.matmul(R_LIDAR_to_LB, ptcloud) + t_LIDAR_to_LB
    # print(type(pt_LB), pt_LB.shape)

    # Projection to spherical image
    pix = xyz2pixel(pt_LB, w, h)
    pix = np.round(pix, 0).astype(np.int)  # round and cast as integer
    return pix


def project_readout(dir_path, readout_idx, rot, tr):
    img_path = join(dir_path, "images", "pano", "image{:04d}.png".format(readout_idx))
    if not os.path.exists(img_path):
        print(img_path, " could not be found")
        return None, None, None

    # Load image
    img = imread(img_path)
    print(img_path)

    ptcloud_path = join(dir_path, "scans", "scan{:04d}.txt".format(readout_idx))
    if not os.path.exists(ptcloud_path):
        print(ptcloud_path, " could not be found")
        return None, None, None

    # Load points and reflectance
    data = []
    with open(ptcloud_path) as f:
        n_lines = read_int(f.readline())
        if n_lines is not None:
            line = True
            while line:
                line = f.readline()
                line_vals = line.split(" ")
                if (len(line_vals) == 4):
                    data.append([float(i) for i in line_vals])
    print("found ", len(data), " points in LiDAR pointcloud")
    print(ptcloud_path)
    data = np.array(data).transpose()

    pt_LIDAR = data[:3, :]
    # reflectance = data[3, :]

    locs_pix = project_pointcloud(img, pt_LIDAR, rot, tr)
    return locs_pix, img, pt_LIDAR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Path to data directory", type=str)
    parser.add_argument("index", help="Index of the camera-lidar capture", type=int)
    parser.add_argument("--indexInterval", help="Index interval length", type=int, default=1)
    parser.add_argument("--yaw", help="Yaw rotation (degrees)", type=float, default=-55.14)
    parser.add_argument("--pitch", help="Pitch rotation (degrees)", type=float, default=15.47)
    parser.add_argument("--roll", help="Roll rotation (degrees)", type=float, default=-0.18)
    parser.add_argument("--tx", help="Tx translation (meters)", type=float, default=0.0934)
    parser.add_argument("--ty", help="Ty translation (meters)", type=float, default=0.0597)
    parser.add_argument("--tz", help="Tz translation (meters)", type=float, default=-0.1659)

    args = parser.parse_args()

    data_path = args.dir  # dirname(abspath(__file__)) "/media/miro/WD/kucl_dataset/indoor" "D:\kucl_dataset\indoor"
    start_idx = args.index

    rotation = [args.yaw, args.pitch, args.roll]
    translation = [args.tx, args.ty, args.tz]

    for i in range(start_idx, start_idx + args.indexInterval):
        locs_pix, img, _ = project_readout(data_path, i, rotation, translation)

        if locs_pix is not None:
            # Visual verification
            # img[pix[1, :], pix[0, :], :3] = 1, 0, 0
            fig = plt.figure()
            plt.imshow(img)
            plt.scatter(x=locs_pix[0, :], y=locs_pix[1, :], marker=',', c='r', lw=0, s=0.1)
            # plt.show()
            fig.tight_layout()
            export_path = join(data_path, "match{:04d}.png".format(i))
            fig.savefig(export_path, dpi=450, bbox_inches='tight')
            print(export_path)
