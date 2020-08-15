import numpy as np
import os

def read_int(input_string):
    try:
        return int(input_string)
    except ValueError:
        return None

ptcloud_path = "/home/miro/stack/nuctech/progress/200810/kucl_29/scan0029.txt"
ptcloud_path = "/home/miro/stack/nuctech/progress/200810/kucl_29/scan0029.txt"
ptcloud_path = "/media/miro/WD/kucl_dataset/indoor/scans"
ptcloud_path = "/media/miro/WD/kucl_dataset/outdoor/scans"

if os.path.isdir(ptcloud_path):
    ptcloud_dir = ptcloud_path

    for ff in os.listdir(ptcloud_dir):
        # print(f.replace("_camera_color_image_raw", ""))
        ptcloud_path1 = os.path.join(ptcloud_dir, ff)
        ptcould_dir1 = os.path.dirname(ptcloud_path1)

        # Load points and reflectance
        data = []
        with open(ptcloud_path1) as f:
            n_lines = read_int(f.readline())
            if n_lines is not None:
                line = True
                while line:
                    line = f.readline()
                    line_vals = line.split(" ")
                    if (len(line_vals) == 4):
                        data.append([float(i) for i in line_vals])

        print("found ", len(data), " points in", ptcloud_path1)
        data = np.array(data)  # .transpose()
        print(data.shape)
        pt_LIDAR = data[:, :3]
        print(pt_LIDAR.shape)
        print(os.path.join(ptcould_dir1, "_" + os.path.basename(ptcloud_path1)))
        np.savetxt(os.path.join(ptcould_dir1, "_" + os.path.basename(ptcloud_path1)), pt_LIDAR, delimiter=' ', fmt='%f')
else:
    if not os.path.exists(ptcloud_path):
        print(ptcloud_path, " could not be found")

    ptcould_dir = os.path.dirname(ptcloud_path)

    # Load points and reflectance
    data = []
    with open(ptcloud_path) as f:
        n_lines = read_int(f.readline())
        if n_lines is not None:
            line = True
            while line:
                line = f.readline()
                line_vals = line.split(" ")
                if (len(line_vals) == 4):
                    data.append([float(i) for i in line_vals])

    print(ptcloud_path)
    print("found ", len(data), " points in LiDAR pointcloud")
    data = np.array(data)#.transpose()
    print(data.shape)
    pt_LIDAR = data[:, :3]
    print(pt_LIDAR.shape)
    np.savetxt(os.path.join(ptcould_dir, "_" + os.path.basename(ptcloud_path)), pt_LIDAR, delimiter=' ', fmt='%f')