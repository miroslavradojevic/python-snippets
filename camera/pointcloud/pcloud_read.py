#!/usr/bin/env python
import open3d as o3d
import numpy as np
import copy

trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4],
                         [0.0, 0.0, 0.0, 1.0]])

# Load point cloud
# http://www.open3d.org/docs/release/tutorial/Basic/file_io.html
# pcd = o3d.io.read_point_cloud("/home/miro/stack/nuctech/progress/200810/l-cas/1464002845.824573000.pcd", format='pcd')
# pcd = o3d.io.read_point_cloud("/home/miro/stack/nuctech/progress/200810/kucl_29/_scan0029.txt", format='xyz')
pcd = o3d.io.read_point_cloud("/media/miro/WD/kucl_dataset/indoor/_scans/_scan0027.txt", format='xyz')
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))

# convert it to a numpy array
pcd_array = np.asarray(pcd.points)
print(pcd_array.shape)
o3d.visualization.draw_geometries([pcd])


pcd1 = o3d.io.read_point_cloud("/media/miro/WD/kucl_dataset/indoor/_scans/_scan0028.txt", format='xyz')
pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=10))

# convert it to a numpy array
pcd_array1 = np.asarray(pcd1.points)
print(pcd_array1.shape)
o3d.visualization.draw_geometries([pcd1])

reg_p2p = o3d.registration.registration_icp(
                    pcd, pcd1, 0.1, trans_init,
                    # o3d.registration.TransformationEstimationPointToPoint(),
                    o3d.registration.TransformationEstimationPointToPlane(),
                    o3d.registration.ICPConvergenceCriteria(max_iteration = 100))

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0]) # [1, 0.706, 0]
    target_temp.paint_uniform_color([0, 0, 1]) # [0, 0.651, 0.929]
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

draw_registration_result(pcd, pcd1, reg_p2p.transformation)

