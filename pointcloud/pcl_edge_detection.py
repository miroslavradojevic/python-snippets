#!/usr/bin/env python

import numpy as np
import argparse
import open3d as o3d
from os.path import exists, splitext, abspath, dirname, join
from utils import edge_detection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pcl_path", help="Path to point cloud file (.pcd | .xyz | .txt)", type=str)
    parser.add_argument("-r", help="Neighborhood: radius", type=float, default=0.15)
    parser.add_argument("-nn", help="Neighborhood: N nearest neighbors", type=int, default=30)
    parser.add_argument("-t", help="Threshold", type=float, default=100)
    args = parser.parse_args()

    # Load points
    if not exists(args.pcl_path):
        exit(args.pcl_path + " could not be found")

    ext = splitext(args.pcl_path)[-1].lower()
    if ext is None or ext not in [".pcd", ".txt", ".xyz"]:
        exit("Point-cloud file has wrong extension")

    # read point cloud
    pcd = o3d.io.read_point_cloud(args.pcl_path, format="pcd" if ext[1:] == "pcd" else "xyz")

    # Method 1
    pcd1, pcd2, pcd3 = edge_detection(pcd, args.r, args.nn, args.t)
    print("{:d} ({:3.2f}%) after edge detection".format(len(pcd1.points), 100.0 * len(pcd1.points) / len(pcd.points)))
    print("{:d} ({:3.2f}%) after edge detection".format(len(pcd2.points), 100.0 * len(pcd2.points) / len(pcd.points)))
    print("{:d} ({:3.2f}%) after edge detection".format(len(pcd3.points), 100.0 * len(pcd3.points) / len(pcd.points)))

    # save
    outdir = dirname(abspath(args.pcl_path))
    fname = splitext(args.pcl_path)[0]
    o3d.io.write_point_cloud(join(outdir, fname + "_edge_detection1.pcd"), pcd1)
    o3d.io.write_point_cloud(join(outdir, fname + "_edge_detection2.pcd"), pcd2)
    o3d.io.write_point_cloud(join(outdir, fname + "_edge_detection3.pcd"), pcd3)
