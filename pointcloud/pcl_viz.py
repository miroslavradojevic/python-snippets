# https://stackoverflow.com/questions/33287156/specify-color-of-each-point-in-scatter-plot-matplotlib
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#scatter-plots
import argparse
import time
from os.path import exists, splitext, isfile, isdir, join, dirname, basename
from os import makedirs, listdir
import open3d as o3d
import numpy as np
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import load_points, get_prefix

def pcd_viz(pcd_path, outdir=None):
    print("pcd_path = [{}]".format(pcd_path))
    print("outdir = [{}]".format(outdir))
    pcd = o3d.io.read_point_cloud(pcd_path, format="pcd") # if ext[1:] == "pcd" else "xyz"

    print(np.array(pcd.points).shape)
    if len(np.array(pcd.points)) == 0:
        print("Could not find points")
        return None

    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Color
    # vis.get_render_option().light_on = False
    vis.get_render_option().point_size = 3.0
    vis.add_geometry(pcd)
    # vis.get_render_option().save_to_json(args.p + "_renderoption.json")
    # buf = vis.capture_screen_float_buffer(do_render=True)
    outpath = join(dirname(pcd_path) if outdir is None else outdir, splitext(basename(pcd_path))[0] + ".jpg")
    vis.capture_screen_image(outpath, do_render=True)

    print(outpath)
    print("")
    # vis.run()
    vis.destroy_window()
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=str, required=True, help="Path to pointcloud file (.pcd | .xyz | .txt)")

    args = parser.parse_args()

    if not exists(args.p):
        exit("{} does not exist".format(args.p))

    if isfile(args.p):
        ext = splitext(args.p)[-1].lower()
        if ext is not None and ext in [".pcd"]:
            pcd_viz(args.p)
        else:
            print("File {} has wrong extension".format(args.p))


    elif isdir(args.p):
        # convert all point cloud files within the directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print(Path(args.p).parent)
        outdir = join(Path(args.p).parent, str(__file__) + "_" + timestamp)

        if not exists(outdir):
            makedirs(outdir)

        for f in listdir(args.p):
            if isfile(join(args.p, f)):
                # check extension
                ext = splitext(f)[-1].lower()
                if ext is not None and ext in [".pcd"]:
                    pcd_viz(join(args.p, f), outdir)
                else:
                    print("skipping {}".format(f))



    # pcd_down = pcd.voxel_down_sample(voxel_size=0.5)
    # o3d.visualization.draw_geometries([pcd_down])
    # print(len(pcd_down.points))

    # Export plot
    # pt = np.asarray(pcd.points).transpose()[:3, :]
    #
    # export_path = get_prefix(args.p) + "_viz.pdf"
    # print(export_path)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pt[0, :], pt[1, :], pt[2, :], marker='o', c="#32CF6980", lw=0.2, s=0.3)
    # plt.show()
    # fig.savefig(export_path, dpi=250, bbox_inches='tight')
