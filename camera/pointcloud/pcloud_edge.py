#!/usr/bin/env python
# from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to point cloud file", type=str)
    args = parser.parse_args()

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = np.random.rand(20,3)
    print(X.shape)

    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(type(indices), indices)
    print(type(distances), distances)