#!/usr/bin/env python
# from sklearn.neighbors import KDTree
# from sklearn.neighbors import NearestNeighbors
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pcl_path", help="Path to pointcloud file (.pcd | .xyz | .txt)", type=str)
    parser.add_argument("r", help="")
    parser.add_argument("nn", help="")
    args = parser.parse_args()



    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # X = np.random.rand(20,3)
    # print(X.shape)
    # nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    # distances, indices = nbrs.kneighbors(X)