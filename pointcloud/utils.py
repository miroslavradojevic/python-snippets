"""Provides helper methods for loading and parsing pointcloud data."""
from collections import namedtuple
from os.path import join, dirname, basename, splitext

import numpy as np
import open3d as o3d

__author__ = "Miroslav Radojevic"
__email__ = "miroslav.radojevic@gmail.com"


def load_points(ptcloud_path, format):
    """
    Expects .xyz .pcd .txt
    Use Open3D library to read based on the file extension
    """
    data = o3d.io.read_point_cloud(ptcloud_path, format=format)
    return data


def get_prefix(file_path):
    return join(dirname(file_path), basename(splitext(file_path)[0]))


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


def filter_zero_points(pts):
    # Filter points that are [0, 0, 0]
    pts_d = np.sqrt(np.multiply(pts[0, :], pts[0, :]) + \
                 np.multiply(pts[1, :], pts[1, :]) + \
                 np.multiply(pts[2, :], pts[2, :]))

    return pts[:, pts_d > 0.0]

def compute_edge_score(pcd, r, nn):
    # Compute set of nearest-neighbor indexes
    # KDTree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    nr_pts = len(pcd.points)
    print("nr_pts", nr_pts, flush=True)
    edge_sc = np.zeros((nr_pts,2))
    for i in range(nr_pts):
        [_, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[i], r, nn)
        p0 = np.asarray(pcd.points)[idx[0], :]
        p0 = np.expand_dims(p0, axis=0)
        pN = np.asarray(pcd.points)[idx[1:], :]
        d_p_Np = p0 - pN

        if len(d_p_Np) > 1:
            edge_score_1_norm = np.amax(np.linalg.norm(d_p_Np, axis=1))
            edge_sc[i, 0] = np.linalg.norm(np.mean(d_p_Np, axis=0)) / edge_score_1_norm if edge_score_1_norm > 0 else 0
            C = np.cov(np.transpose(pN)) # Estimate a covariance matrix, Each row of m represents a variable, and each column a single observation
            w, _ = np.linalg.eig(C)
            # https://stackoverflow.com/questions/10083772/python-numpy-sort-eigenvalues
            w.sort()
            if w[2] > 0:
                edge_sc[i, 1] = 1 - ((w[1]-w[0])/w[2])
            else:
                edge_sc[i, 1] = 0
        else:
            edge_sc[i, :] = np.zeros((1, 2))
    return edge_sc