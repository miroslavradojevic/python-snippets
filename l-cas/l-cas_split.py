#!/usr/bin/env python
import argparse
from os.path import isdir, splitext, exists, join
from os import listdir, makedirs
import numpy as np
import shutil
import time
from pathlib import Path

def read_float(s):
    try:
        return float(s)
    except ValueError:
        return None

def read_int(s):
    try:
        return int(s)
    except ValueError:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split list of.pcd files into 2 directories with annotations (.txt) and without")
    parser.add_argument("-d", type=str, required=True, help="Path to directory with point cloud files")
    parser.add_argument("-cam", type=str, default=None, help="Path to directory with jpgs")
    args = parser.parse_args()

    if not isdir(args.d):
        print("Error: {} is not a directory".format(args.d))
        exit()

    picture_names = None
    picture_tstamps = None
    if args.cam is not None and exists(args.cam) and isdir(args.cam):
        picture_names = listdir(args.cam)
        picture_names = [x for x in picture_names if splitext(x)[-1].lower() == ".jpg"]
        # print(picture_names, len(picture_names))
        picture_tstamps = [read_float(x.replace(".jpg", "")) for x in picture_names]
        # print(picture_tstamps, type(picture_tstamps[0]), type(picture_tstamps))
        picture_tstamps = np.array(picture_tstamps)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = join(Path(args.d).parent, str(__file__) + "_" + timestamp)

    if not exists(outdir):
        makedirs(outdir)
        makedirs(join(outdir, "free"))
        makedirs(join(outdir, "obstacle"))

    f_list = listdir(args.d)
    count = {"free": 0, "obstacle": 0}
    for f in listdir(args.d):
        ext = splitext(f)[-1].lower()
        if ext in [".pcd"]:
            f_tstamp = f.replace(ext, "")

            print(f, flush=True)

            annot_name = f.replace(ext, ".txt")
            category = "obstacle" if annot_name in f_list else "free"
            print(category, flush=True)
            count[category] += 1

            # copy pcd
            print(join(args.d, f), "---", join(outdir, category, f_tstamp + ".pcd"))
            shutil.copy(join(args.d, f), join(outdir, category, f_tstamp + ".pcd"))

            if annot_name in f_list:
                # copy txt
                print(join(args.d, annot_name), "---", join(outdir, category, f_tstamp + ".txt"))
                shutil.copy(join(args.d, annot_name), join(outdir, category, f_tstamp + ".txt"))

            if picture_tstamps is not None:
                # find nearest timestamp
                f_stamp_flt = read_float(f_tstamp)
                d_tstamp = np.abs(f_stamp_flt - picture_tstamps)
                img_idx = np.argmin(d_tstamp)
                # copy jpg
                print(join(args.cam, picture_names[img_idx]), "---", join(outdir, category, f_tstamp + ".jpg"))
                shutil.copy(join(args.cam, picture_names[img_idx]), join(outdir, category, f_tstamp + ".jpg"))
                # matches = [x for x in lst if fulfills_some_condition(x)]

    print(count)
