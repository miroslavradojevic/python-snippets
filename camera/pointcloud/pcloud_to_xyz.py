import numpy as np
import os

def read_int(input_string):
    try:
        return int(input_string)
    except ValueError:
        return None

ptcloud_path = "/home/miro/stack/nuctech/progress/200810/kucl_29/scan0029.txt"
ptcloud_path = "/home/miro/stack/nuctech/progress/200810/kucl_29/scan0029.txt"

ptcould_dir = os.path.dirname(ptcloud_path)


if not os.path.exists(ptcloud_path):
    print(ptcloud_path, " could not be found")

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
print("found ", len(data), " points in LiDAR pointcloud")
print(ptcloud_path)
data = np.array(data)#.transpose()
print(data.shape)
pt_LIDAR = data[:, :3]
print(pt_LIDAR.shape)
print(os.path.join(ptcould_dir, "_" + os.path.basename(ptcloud_path)))
np.savetxt(os.path.join(ptcould_dir, "_" + os.path.basename(ptcloud_path)), pt_LIDAR, delimiter=' ', fmt='%f')