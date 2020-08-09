#!/usr/bin/env python
import open3d as o3d
import numpy as np

# Load point cloud
# http://www.open3d.org/docs/release/tutorial/Basic/file_io.html
# pcd = o3d.io.read_point_cloud("/home/miro/stack/nuctech/progress/200810/l-cas/1464002845.824573000.pcd", format='pcd')
pcd = o3d.io.read_point_cloud("/home/miro/stack/nuctech/progress/200810/kucl_29/_scan0029.txt", format='xyz')

# convert it to a numpy array
pcd_array = np.asarray(pcd.points)

# view point cloud
o3d.visualization.draw_geometries([pcd])

pcd_tree = o3d.geometry.KDTreeFlann(pcd)


