import os

dir_path = "/media/miro/WD/kitti05"

dlist = os.listdir(dir_path)

for f in dlist:
    if os.path.splitext(f)[-1].lower() == ".pcd":
        f_new = f[:10] + "." + f[10:]
        path_old = os.path.join(dir_path, f)
        path_new = os.path.join(dir_path, f_new)
        os.rename(path_old, path_new)
        print(path_old, "\n")
        print(path_new, "\n")

