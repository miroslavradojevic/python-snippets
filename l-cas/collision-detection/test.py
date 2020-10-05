#!/usr/bin/env python
import argparse
import math
import time
import cv2
import open3d as o3d
import numpy as np
from os.path import isdir, join, exists, splitext
from os import listdir, makedirs
from numpy import linalg as LA
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def pcd_to_sphproj(pcd_path, nr_scans, width, outdir=None):
    pcd = o3d.io.read_point_cloud(pcd_path, format="pcd")
    pcd_arr = np.asarray(pcd.points)

    if len(pcd_arr) == 0:
        return None

    # https://towardsdatascience.com/spherical-projection-for-point-clouds-56a2fc258e6c

    # print(pcd_arr.shape)
    # print(pcd_arr[:, :3].shape)

    R = LA.norm(pcd_arr[:, :3], axis=1)
    # print("R {} | {} -- {}".format(R.shape, np.amin(R), np.amax(R)))

    yaw = np.arctan2(pcd_arr[:, 1], pcd_arr[:, 0])
    # print("yaw {} | {} -- {}".format(yaw.shape, np.amin(yaw), np.amax(yaw)))
    # print("y {} | {} -- {}".format(pcd_arr[:, 1].shape, np.amin(pcd_arr[:, 1]), np.amax(pcd_arr[:, 1])))

    pitch = np.arcsin(np.divide(pcd_arr[:, 2], R))
    # print("pitch {} | {} -- {}".format(pitch.shape, np.amin(pitch), np.amax(pitch)))

    FOV_Down = np.amin(pitch)
    FOV_Up = np.amax(pitch)
    FOV = FOV_Up + abs(FOV_Down)

    u = np.around((nr_scans - 1) * (1.0 - (pitch - FOV_Down) / FOV)).astype(np.int16)
    # print("u {} | {} -- {} | {}".format(u.shape, np.amin(u), np.amax(u), u.dtype))

    v = np.around((width - 1) * (0.5 * ((yaw / math.pi) + 1))).astype(np.int16)
    # print("v {} | {} -- {} | {}".format(v.shape, np.amin(v), np.amax(v), v.dtype))

    sph_proj = np.zeros((nr_scans, width))

    R[R > 100.0] = 100.0  # cut off all values above 100m
    R = np.round((R / 100.0) * 255.0)  # convert 0.0-100.0m into 0.0-255.0 for saving as byte8 image

    sph_proj[u, v] = R

    # print("sph_proj {} | {} -- {} | {}".format(sph_proj.shape, np.amin(sph_proj), np.amax(sph_proj), sph_proj.dtype))

    sph_proj = sph_proj.astype(np.uint8)  # to be equivallent to saving into 8-bit image and reading the same image

    # outpath = join(Path(pcd_path).parent if outdir is None else outdir, splitext(basename(pcd_path))[0] + ".jpg")
    # cv2.imwrite(outpath, sph_proj)
    # print(outpath)
    return np.expand_dims(sph_proj, -1)  # np.amin(R), np.amax(R)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", type=str, required=True, help="Data directory with .pcd files")
    parser.add_argument("-model", type=str, required=True, help="Path to .h5 model")
    args = parser.parse_args()

    if not exists(args.d):
        exit("{} does not exist".format(args.d))

    if not isdir(args.d):
        exit("{} is not a directory".format(args.d))

    model = load_model(args.model)  # , custom_objects={'AttentionLayer': AttentionLayer}
    model.summary()
    nr_layers = model.layers[0].input_shape[1]
    scan_width = model.layers[0].input_shape[2]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = join(Path(args.d).parent, str(__file__) + "_" + timestamp)

    if not exists(outdir):
        makedirs(outdir)

    dir_list = listdir(args.d)

    for f in dir_list:
        # check extension
        ext = splitext(f)[-1].lower()

        if ext == ".pcd":
            path_pointcloud = join(args.d, f)
            path_camera = join(args.d, f.replace(ext, ".jpg"))

            if exists(path_camera):
                image_camera = cv2.imread(path_camera)
                scale_percent = 50  # percent of original size
            width = int(image_camera.shape[1] * scale_percent / 100)
            height = int(image_camera.shape[0] * scale_percent / 100)
            dim = (width, height)
            # resize image
            image_camera = cv2.resize(image_camera, dim, interpolation=cv2.INTER_CUBIC)
            # print("cam: {} {} {}".format(type(image_camera), image_camera.shape, image_camera[0].dtype))

            sph_proj = pcd_to_sphproj(path_pointcloud, nr_layers, scan_width)
            sph_proj = sph_proj / 255.0
            # print("sphproj: {} {} {}".format(type(sph_proj), sph_proj.shape, sph_proj[0].dtype))
            sph_proj_image = np.expand_dims(sph_proj, axis=0)
            pred = model.predict(sph_proj_image)
            # pred: {"0": "free", "1": "obstacle"}
            pred_idx = np.argmax(pred)

            if pred_idx == 0:
                # free - add green circle to the camera image
                indicator_color = (0, 255, 0)
            elif pred_idx == 1:
                # obstacle - add red circle
                indicator_color = (0, 0, 255)

            cv2.circle(img=image_camera, center=(width // 2, height // 2), radius=width // 8, color=indicator_color,
                       thickness=-1)

            # plt.imshow(image_camera)
            cv2.imwrite(join(outdir, f.replace(ext, ".jpg")), image_camera)
