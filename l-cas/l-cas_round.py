#!/usr/bin/env python
import argparse
from os.path import isdir, splitext, exists, join, dirname
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", type=str, required=True, help="Path to directory with point cloud files")

    args = parser.parse_args()

    if not exists(args.d):
        exit("{} does not exist".format(args.d))

    if not isdir(args.d):
        exit("{} is not a directory".format(args.d))

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = join(Path(args.d).parent, str(__file__) + "_" + timestamp)

    if not exists(outdir):
        makedirs(outdir)

    for f in listdir(args.d):
        # check extension
        ext = splitext(f)[-1].lower()
        if ext in [".pcd"]:
            f_tstamp = f.replace(ext, "")
            f_stamp_flt = read_float(f_tstamp)
            if f_stamp_flt is not None:
                f_stamp_int = int(f_stamp_flt) # round(f_stamp_flt, 0)

                source_pcd = join(args.d, f)
                target_pcd = join(outdir, str(f_stamp_int) + ext)

                if not exists(target_pcd):
                    print("source: {}\ntarget: {}\n".format(source_pcd, target_pcd))
                    shutil.copy(source_pcd, target_pcd)
