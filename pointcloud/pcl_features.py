#!/usr/bin/env python
# from sklearn.neighbors import NearestNeighbors
import numpy as np
import argparse
from os.path import exists, splitext
import open3d as o3d
from utils import load_points, compute_edge_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pcl_path", help="Path to pointcloud file (.pcd | .xyz | .txt)", type=str)
    parser.add_argument("-r", help="Neighborhood radius", type=float, default=3.0)
    parser.add_argument("-nn", help="Neighborhood - N nearest neighbors", type=int, default=20)
    args = parser.parse_args()

    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # X = np.random.rand(20,3)
    # nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    # distances, indices = nbrs.kneighbors(X)

    # Load points
    if not exists(args.pcl_path):
        exit(args.pcl_path + " could not be found")

    ext = splitext(args.pcl_path)[-1].lower()
    if ext is None or ext not in [".pcd", ".txt", ".xyz"]:
        exit("Point-cloud file has wrong extension")

    # http://www.open3d.org/docs/release/tutorial/Basic/file_io.html
    pcd = load_points(args.pcl_path, "pcd" if ext[1:] == "pcd" else "xyz")
    # o3d.visualization.draw_geometries([pcd])
    print(len(pcd.points), "points found")

    # pcd_array = np.asarray(pcd.points)



    # search_knn_vector_3d - returns a list of indices of the k nearest neighbors of the anchor point
    # [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[0], args.nn)
    # print(np.asarray(pcd.points)[idx[1:], :])

    # search_radius_vector_3d - all points with distances to the anchor point less than a given radius
    # [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[0], args.r)
    # print(np.asarray(pcd.points)[idx[1:], :])

    # search_hybrid_vector_3d - at most k nearest neighbors that have distances to the anchor point less than a given radius
    # Combines criteria of KNN search and RNN search
    # [_, idx, _] = pcd_tree.search_hybrid_vector_3d(pcd.points[0], args.r, args.nn)
    # np.asarray(pcd.points)[idx[1:], :]

    pcd_edge_score = compute_edge_score(pcd, args.r, args.nn)
    # Normalize each score
    pcd_edge_score = pcd_edge_score / pcd_edge_score.max(axis=0)
    # Multiply scores
    pcd_edge_score = np.prod(pcd_edge_score, axis=1)

    # give all points grey color
    pcd.paint_uniform_color([0.0, 0.0, 0.0])

    # Set colors based on the score
    from matplotlib import cm
    reds = cm.get_cmap('Reds')
    np.asarray(pcd.colors)[:, :] = reds(pcd_edge_score)[:, :3]

    o3d.visualization.draw_geometries([pcd])

