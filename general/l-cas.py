#!/usr/bin/env python
import argparse
from os.path import isdir, splitext
from os import listdir

def read_float(s):
    try:
        return float(s)
    except ValueError:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="Path to directory with point cloud files", type=str)
    # parser.add_argument("-r", help="Neighborhood: radius", type=float, default=0.15)
    # parser.add_argument("-nn", help="Neighborhood: N nearest neighbors", type=int, default=30)
    # parser.add_argument("-t", help="Threshold", type=float, default=100)
    args = parser.parse_args()

    if not isdir(args.d):
        print("Error: argument [{}] is not a directory".format(args.d))
        exit()

    f_list = listdir(args.d)
    # jpg_list =
    print(type(f_list))

    for f in listdir(args.d):
        f_ext = splitext(f)[-1].lower()
        if f_ext in [".pcd"]:
            # exit("Point-cloud file has wrong extension")
            f_tstamp = f.replace(f_ext, "")
            t1 = read_float(f_tstamp)
            # matches = [x for x in lst if fulfills_some_condition(x)]

            print(f_tstamp, t1, type(f_list), "{}.txt".format(f_tstamp) in f_list)




