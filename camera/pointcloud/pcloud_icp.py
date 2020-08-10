#!/usr/bin/env python
import argparse
from os.path import isdir, join, splitext
from os import listdir
import open3d as o3d
import numpy as np
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def draw_registration_result(src, tgt, transformation):
    source_temp = copy.deepcopy(src)
    target_temp = copy.deepcopy(tgt)
    source_temp.paint_uniform_color([1, 0, 0]) # [1, 0.706, 0]
    target_temp.paint_uniform_color([0, 0, 1]) # [0, 0.651, 0.929]
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # zoom = 0.4459,
    # front = [0.9288, -0.2951, -0.2242],
    # lookat = [1.6784, 2.0612, 1.4451],
    # up = [-0.3402, -0.9189, -0.1996]


# print("test")
# aa = o3d.io.read_point_cloud("/media/miro/WD/kucl_dataset/indoor/_scans/_scan0029.txt", format='xyz')  # (target_path, format=format)
# print("aa=", np.asarray(aa.points).shape)
# if True:
#     exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pcl", help="Path to piont cloud file/directory", type=str)
    parser.add_argument("--pcl2", help="Path to piont cloud file", type=str, default=None)
    parser.add_argument("--radius", help="Radius used to compute normal vector", type=float, default=0.1)
    parser.add_argument("--nn_max", help="Max NN used to compute normal vector", type=int, default=30)
    parser.add_argument("--thr", help="Max NN used to compute normal vector", type=float, default=0.02)
    args = parser.parse_args()

    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4],
                             [0.0, 0.0, 0.0, 1.0]])

    # trans_init = np.asarray([[0, 0, 0, 0.5],
    #                          [0, 0, 0, 0.5],
    #                          [0, 0, 0, 0.0],
    #                          [0, 0, 0, 1.0]])

    if isdir(args.pcl):
        t = []
        source = None
        for file_pcl in sorted(listdir(args.pcl)):
            ext = splitext(file_pcl)[-1].lower()
            if ext == ".pcd" or ext == ".txt" and True:
                format = "pcd" if ext[1:] == "pcd" else "xyz"

                file_curr = join(args.pcl, file_pcl)

                if source is None:
                    print(file_curr, format)
                    source = o3d.io.read_point_cloud(file_curr, format=format)
                    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.radius, max_nn=args.nn_max))
                else:
                    print(file_curr, format)
                    target = o3d.io.read_point_cloud(file_curr, format=format)
                    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.radius, max_nn=args.nn_max))

                    # register source-target
                    reg_p2l = o3d.registration.registration_icp(
                        source, target, args.thr, trans_init,
                        o3d.registration.TransformationEstimationPointToPlane(),
                        o3d.registration.ICPConvergenceCriteria(max_iteration = 50))
                    # print(reg_p2l)
                    # print("Transformation is:", reg_p2l.transformation)
                    t.append(reg_p2l.transformation[:3,3].transpose())

                    source = target # recursion
                    trans_init = reg_p2l.transformation

        t = np.array(t)
        t = np.cumsum(t, axis=0)
        print(t.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(t[:,0], t[:,1], t[:,2]) # , c=y
        plt.show()

    elif args.pcl2 is not None:
        source_path = args.pcl
        target_path = args.pcl2

        source_ext = splitext(source_path)[-1].lower()
        target_ext = splitext(target_path)[-1].lower()

        if source_ext == ".pcd" or source_ext == ".txt":
            format = "pcd" if source_ext[1:] == "pcd" else "xyz"

            print(source_path, format)
            print(type(source_path), type(format))
            pcd_src = o3d.io.read_point_cloud(source_path, format=format)# ("/media/miro/WD/kucl_dataset/indoor/_scans/_scan0028.txt", format='xyz') #
            print(np.asarray(pcd_src.points).shape)
            pcd_src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.radius, max_nn=args.nn_max))

            if target_ext == ".pcd" or target_ext == ".txt":
                format = "pcd" if target_ext[1:] == "pcd" else "xyz"

                print(target_path, format)
                pcd_tgt = o3d.io.read_point_cloud(target_path, format=format)# ("/media/miro/WD/kucl_dataset/indoor/_scans/_scan0029.txt", format='xyz') #
                print(np.asarray(pcd_tgt.points).shape)
                pcd_tgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.radius, max_nn=args.nn_max))

                reg_p2l = o3d.registration.registration_icp(
                    pcd_src, pcd_tgt, args.thr, trans_init,
                    o3d.registration.TransformationEstimationPointToPlane(),
                    o3d.registration.ICPConvergenceCriteria(max_iteration = 50))

                print("Transformation is:", reg_p2l.transformation)
                print(type(reg_p2l.transformation), reg_p2l.transformation.shape)

                draw_registration_result(pcd_src, pcd_tgt, reg_p2l.transformation)

