#!/usr/bin/env python
import open3d as o3d
import numpy as np

# Load point cloud
# http://www.open3d.org/docs/release/tutorial/Basic/file_io.html
# pcd = o3d.io.read_point_cloud("/home/miro/stack/nuctech/progress/200810/l-cas/1464002845.824573000.pcd", format='pcd')
pcd = o3d.io.read_point_cloud("/home/miro/stack/nuctech/progress/200810/kucl_29/_scan0029.txt", format='xyz')

# give all points grey color
pcd.paint_uniform_color([0.5, 0.5, 0.5])

# convert it to a numpy array
pcd_array = np.asarray(pcd.points)

# KDTree
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

nr_points = pcd_array.shape[0]

# Paint the random point red
point_idx = np.random.randint(nr_points)
pcd.colors[point_idx] = [1, 0, 0]

# use search_knn_vector_3d
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[point_idx], 3)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

# visualize point cloud
o3d.visualization.draw_geometries([pcd])

# search_radius_vector_3d
[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[point_idx], 0.2)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

# search_hybrid_vector_3d() -- RKNN
# Returns at most k nearest neighbors that have distances to the anchor point less than a given radius
# Combines criteria of KNN search and RNN search

