import os

dir_path = "/home/miro/stack/nuctech/progress/200629/0014.bag"

dd = os.listdir(dir_path)
# print(len(dd))

for f in dd:
    # print(f.replace("_camera_color_image_raw", ""))
    path_old = os.path.join(dir_path, f)
    name_new = f.replace("_camera_color_image_raw", "").replace(".png", "")
    name_new = "{:.3f}.png".format(float(name_new))
    # print("{0} -> {1:.2f}".format(name_new_float, name_new_float))
    path_new = os.path.join(dir_path, name_new)
    print(path_old, "\n", path_new, "\n")
    os.rename(path_old, path_new)

