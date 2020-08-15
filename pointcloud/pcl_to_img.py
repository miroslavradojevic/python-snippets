import argparse
import math
from collections import namedtuple
from os.path import exists, splitext, join, dirname, basename, isfile

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.image import imread
from scipy.spatial.transform import Rotation as R


def get_prefix(file_path):
    return join(dirname(file_path), basename(splitext(file_path)[0]))


def load_points(ptcloud_path, format):
    """
    Expects two formats .xyz or .pcd
    Use Open3D library to read based on the file extension
    """
    data = o3d.io.read_point_cloud(ptcloud_path, format=format)
    return data


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


def project_pointcloud_ladybug5(im, ptcloud, rot, tr):
    # Rigid-body transformation between LiDAR and camera
    R_lidar_to_lb = R.from_euler('ZYX', np.deg2rad(rot)).as_dcm()  # Rotation: [yaw, pitch, roll] in degrees
    t_lidar_to_lb = np.array(tr).reshape(3, 1)  # Translation [tx, ty, tz] in meters

    w = im.shape[1]
    h = im.shape[0]

    # Transform points w.r.t. LIDAR coordinate frame to Ladybug5 coordinate frame
    pt_lb = np.matmul(R_lidar_to_lb, ptcloud) + t_lidar_to_lb

    # Projection to spherical image
    pix = xyz2pixel(pt_lb, w, h)
    pix = np.round(pix, 0).astype(np.int)  # round and cast as integer
    return pix


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def load_calib(calib_filepath):
    """Load and compute intrinsic and extrinsic calibration parameters."""
    # We'll build the calibration parameters as a dictionary, then
    # convert it to a namedtuple to prevent it from being modified later
    data = {}

    # Load the calibration file
    # calib_filepath = os.path.join(self.sequence_path, 'calib.txt')

    filedata = read_calib_file(calib_filepath)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
    data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
    data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
    data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
    data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the stereo baselines in meters by projecting the origin of
    # each camera frame into the velodyne frame and computing the distances
    # between them
    p_cam = np.array([0, 0, 0, 1])
    p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
    p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
    p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
    p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

    data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
    data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)  # rgb baseline

    return namedtuple('CalibData', data.keys())(*data.values())


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

    if not exists(args.img):
        exit(args.img + " could not be found")

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
    img = imread(args.img)

    # Load points
    if not exists(args.pcl):
        exit(args.pcl + " could not be found")

    ext = splitext(args.pcl)[-1].lower()
    if ext is None or ext not in [".pcd", ".txt", ".xyz"]:
        exit("Point-cloud file has wrong extension")

    pts = load_points(args.pcl, "pcd" if ext[1:] == "pcd" else "xyz")
    pt_lidar = np.asarray(pts.points).transpose()[:3, :]

    # Filter points that are [0, 0, 0]
    rr = np.sqrt(np.multiply(pt_lidar[0, :], pt_lidar[0, :]) + \
                 np.multiply(pt_lidar[1, :], pt_lidar[1, :]) + \
                 np.multiply(pt_lidar[2, :], pt_lidar[2, :]))

    pt_lidar = pt_lidar[:, rr > 0.0]

    if yaw is not None:
        locs_pix = project_pointcloud_ladybug5(img, pt_lidar, [yaw, pitch, roll], [tx, ty, tz])
    elif calib is not None:
        locs_pix = project_pointcloud_kitti(img, pt_lidar, calib)

    if locs_pix is not None:
        # Visual verification
        fig = plt.figure()
        plt.imshow(img)
        plt.scatter(x=locs_pix[0, :], y=locs_pix[1, :], marker='o', c='#f5784280', lw=0, s=0.3)
        fig.tight_layout()
        export_path = get_prefix(args.img) + "_" + basename(splitext(args.pcl)[0]) + "_" + (
            args.t if calib is None else "calib.txt") + ".pdf"
        fig.savefig(export_path, dpi=150, bbox_inches='tight')
        print(export_path)
