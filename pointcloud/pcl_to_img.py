import argparse
import math
import sys
from os.path import dirname, abspath
from os.path import exists, splitext, basename, isfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from scipy.spatial.transform import Rotation as R

from utils import load_points, get_prefix, load_calib, compute_edge_score

print(abspath(__file__))
print(dirname(abspath(__file__)))
sys.path.append(dirname(dirname(abspath(__file__))))
from edge_det.imedge import edge_detection, edge_detection_1, edge_detection_2


def xyz2pixel(pt, w, h):
    # Range from camera to points
    r = np.sqrt(np.multiply(pt[0, :], pt[0, :]) + \
                np.multiply(pt[1, :], pt[1, :]) + \
                np.multiply(pt[2, :], pt[2, :]))

    # uv of points
    th = np.arctan2(pt[1, :], pt[0, :])
    phi = np.arcsin(np.divide(pt[2, :], r))
    u = -1 / (2 * math.pi) * th + 0.5
    v = -1 / math.pi * phi + 0.5
    # Pixels of points
    pixel = np.vstack((u * w, v * h))
    return pixel


def project_pointcloud_kucl(im, ptcloud, rot, tr):
    # Rigid-body transformation between LiDAR and camera
    R_lidar_to_lb = R.from_euler('ZYX', np.deg2rad(rot)).as_dcm()  # Rotation: [yaw, pitch, roll] in degrees
    t_lidar_to_lb = np.array(tr).reshape(3, 1)  # Translation [tx, ty, tz] in meters
    print(R_lidar_to_lb)
    w = im.shape[1]
    h = im.shape[0]

    # Transform points w.r.t. LIDAR coordinate frame to Ladybug5 coordinate frame
    pt_lb = np.matmul(R_lidar_to_lb, ptcloud) + t_lidar_to_lb

    # Projection to spherical image
    pix = xyz2pixel(pt_lb, w, h)
    pix = np.round(pix, 0).astype(np.int)  # round and cast as integer
    return pix


def project_pointcloud_kitti(im, ptcloud, calib):
    # Rigid-body transformation between LiDAR and camera
    ptcloud = np.vstack([ptcloud, np.ones((1, ptcloud.shape[1]))])
    pt_cam = calib.T_cam2_velo.dot(ptcloud)  # np.matmul(calib.T_cam2_velo, ptcloud)
    print("T_cam2_velo:\n", calib.T_cam2_velo)
    print("P_rect_20:\n", calib.P_rect_20)

    # Projection to image
    pt_uvw = calib.P_rect_20.dot(pt_cam)
    pix = np.vstack([np.divide(pt_uvw[0, :], pt_uvw[2, :]), np.divide(pt_uvw[1, :], pt_uvw[2, :])])
    # pix = np.round(pix, 0).astype(np.int)

    w = im.shape[1]
    h = im.shape[0]

    # print("pix:\n", pix.shape)
    pix = pix[:, pix[0, :] >= 0]
    pix = pix[:, pix[1, :] >= 0]

    pix = pix[:, pix[0, :] < w]
    pix = pix[:, pix[1, :] < h]

    print("pix:\n", pix.shape)
    return pix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        help='Transformation as comma delimited string: yaw,pitch,roll,tx,ty,tz or path to calib.txt',
                        type=str,
                        default="-55.14,15.47,-0.18,0.0934,0.0597,-0.1659")
    parser.add_argument("pcl_path", help="Path to pointcloud file (.pcd | .xyz | .txt)", type=str)
    parser.add_argument("img_path", help="Path to image (.jpg, .png)", type=str)
    args = parser.parse_args()

    if args.t is None:
        exit("Transformation was not given")

    if not exists(args.img_path):
        exit(args.img_path + " could not be found")

    yaw, pitch, roll, tx, ty, tz = [None] * 6
    calib = None

    if isfile(args.t) and basename(args.t) == "calib.txt":
        calib = load_calib(args.t)
        # calib.P_rect_20
        # 3x4 project 3D points in the camera coordinate frame to 2D pixel coordinates
        # calib.T_cam2_velo, velodyne to rectified camera coordinate transforms
    elif len(args.t.split(',')) == 6:
        yaw, pitch, roll, tx, ty, tz = [float(item) for item in args.t.split(',')]
    else:
        exit("There need to be 6 comma-delimited values in transformation or path to the calib.txt file")

    # Load image
    img = imread(args.img_path)

    # Load image edges
    # edges = edge_detection(args.img_path, 4.0, 0.04, 50, 200)
    # edges = edge_detection_1(args.img_path, 3, 50, 70)
    edges = edge_detection_2(args.img_path, 3)
    print("edges:", edges.shape, type(edges), edges[0].dtype, np.amin(edges), np.amax(edges))

    # Load points
    if not exists(args.pcl_path):
        exit(args.pcl_path + " could not be found")

    ext = splitext(args.pcl_path)[-1].lower()
    if ext is None or ext not in [".pcd", ".txt", ".xyz"]:
        exit("Point-cloud file has wrong extension")

    pts = load_points(args.pcl_path, "pcd" if ext[1:] == "pcd" else "xyz")
    pts_array = np.asarray(pts.points).transpose()[:3, :]

    # pts_array = filter_zero_points(pts_array)

    # Compute pointcloud edge features
    pts_edge_score = compute_edge_score(pts, 5.0, 100)
    pts_edge_score = pts_edge_score / pts_edge_score.max(axis=0)
    pts_edge_score = np.prod(pts_edge_score, axis=1)

    if yaw is not None:
        locs_pix = project_pointcloud_kucl(img, pts_array, [yaw, pitch, roll], [tx, ty, tz])
    elif calib is not None:
        locs_pix = project_pointcloud_kitti(img, pts_array, calib)

    if locs_pix is not None:
        # Visual verification
        fig = plt.figure()
        plt.imshow(img)
        plt.scatter(x=locs_pix[0, :], y=locs_pix[1, :], marker='o', c='#f5784280', lw=0, s=pts_edge_score * 2.0)
        fig.tight_layout()
        export_path = get_prefix(args.img_path) + "_" + basename(splitext(args.pcl_path)[0]) + "_" + (
            args.t if calib is None else "calib.txt") + ".pdf"
        fig.savefig(export_path, dpi=150, bbox_inches='tight')
        print(export_path)

        # Overlay over gradient image
        fig = plt.figure()
        plt.imshow(edges, cmap='gray') #  , vmin=0, vmax=255
        plt.scatter(x=locs_pix[0, :], y=locs_pix[1, :], marker='o', c='#f5784280', lw=0, s=pts_edge_score * 2.0)
        # fig.tight_layout()
        export_path = get_prefix(args.img_path) + "_edge" + "_" + basename(splitext(args.pcl_path)[0]) + "_" + (
            args.t if calib is None else "calib.txt") + ".pdf"
        fig.savefig(export_path, dpi=150, bbox_inches='tight')
        print(export_path)
