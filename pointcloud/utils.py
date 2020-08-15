"""Provides helper methods for loading and parsing pointcloud data."""
import open3d as o3d
from os.path import join, dirname, basename, splitext

__author__ = "Miroslav Radojevic"
__email__ = "miroslav.radojevic@gmail.com"


def load_points(ptcloud_path, format):
    """
    Expects .xyz .pcd .txt
    Use Open3D library to read based on the file extension
    """
    data = o3d.io.read_point_cloud(ptcloud_path, format=format)
    return data

def get_prefix(file_path):
    return join(dirname(file_path), basename(splitext(file_path)[0]))

