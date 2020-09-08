# https://stackoverflow.com/questions/33287156/specify-color-of-each-point-in-scatter-plot-matplotlib
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#scatter-plots
import argparse
from os.path import exists, splitext
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from utils import load_points, get_prefix

if __name__ == '__main__':
    # o3d.io.read_point_cloud("/media/miro/WD/kucl_dataset/indoor/_scans/_scan0029.txt", format='xyz')  # (target_path, format=format)
    parser = argparse.ArgumentParser()
    parser.add_argument("pcl_path", help="Path to pointcloud file (.pcd | .xyz | .txt)", type=str)
    args = parser.parse_args()

    # Load points
    if not exists(args.pcl_path):
        exit(args.pcl_path + " could not be found")

    ext = splitext(args.pcl_path)[-1].lower()
    if ext is None or ext not in [".pcd", ".txt", ".xyz"]:
        exit("Point-cloud file has wrong extension")

    pcd = load_points(args.pcl_path, "pcd" if ext[1:] == "pcd" else "xyz")
    o3d.visualization.draw_geometries([pcd])

    # pcd_down = pcd.voxel_down_sample(voxel_size=0.5)
    # o3d.visualization.draw_geometries([pcd_down])
    # print(len(pcd_down.points))

    pt = np.asarray(pcd.points).transpose()[:3, :]
    print(pt.shape)

    export_path = get_prefix(args.pcl_path) + "_viz.pdf"
    print(export_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pt[0, :], pt[1, :], pt[2, :], marker='o', c="#32CF6980", lw=0.2, s=0.3)
    plt.show()
    fig.savefig(export_path, dpi=250, bbox_inches='tight')
