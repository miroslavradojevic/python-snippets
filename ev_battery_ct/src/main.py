#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import SimpleITK as sitk
import cv2
from os.path import exists, splitext, join
from os import makedirs
from time import strftime
from math import inf

from image.io import raw_to_npy
from tools.detect import extract_cells

if __name__=='__main__':
    psr = argparse.ArgumentParser(description='EV battery image analysis')
    psr.add_argument('-m', type=str, required=True, help='Select method')
    psr.add_argument('-f', type=str, required=True, help='Path to the input image file (image stack, binary raw)')
    psr.add_argument('-sz', required=False, nargs='+', help="(width,height,length)", type=int)
    psr.add_argument("-big_endian", required=False, type=lambda v: v.lower() in {"1", "true"}, default=False, help="\'false\' for little-endian or \'true\' for big-endian")
    psr.add_argument('-type', required=False, default="sitkFloat32", help="SimpleITK pixel type (default: sitkFloat32)")
    psr.add_argument('-min_val', required=False, default=0.00, help="Cropping range: min value")
    psr.add_argument('-max_val', required=False, default=0.05, help="Cropping range: max value")

    args = psr.parse_args()

    method = args.m.upper()

    if any(d is None for d in (args.f, args.sz, args.big_endian, args.type, args.min_val, args.max_val)):
        sys.exit("RAW image parameters are found to be missing")

    if not exists(args.f):
        sys.exit(f"File {args.f} could not be found")
        
    if splitext(args.f)[1].upper() !=  ".RAW":
        sys.exit(f"File extension must be .raw")

    if method == "RAW2TIF":
        raw_to_tif(args.f, args.sz, args.big_endian, args.type)# , 0.00, 0.05      
    elif method == "RAW2NPY":
        img = raw_to_npy(args.f, args.sz, args.big_endian, args.type, 0.00, 0.05)
    elif method == "BATTERY":
        img = raw_to_npy(args.f, args.sz, args.big_endian, args.type, 0.00, 0.05)
        img = np.median(img,0)
        
        x1,x2,y1,y2 = detect_battery_rectangle(img)
        
        # crop the rectangle
        img = img[y1:y2, x1:x2]
        
        # visualize cropped battery
        img_viz = np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(uint8)
        cv2.imwrite("battery_crop.jpg", img_viz)
        
    elif method=="CELLS":
        img3d = raw_to_npy(args.f, args.sz, args.big_endian, args.type, args.min_val, args.max_val)
        img = np.mean(img3d, 0) # z-projection, median or mean
        cell_centroids, patch_width, patch_height = extract_cells(img, img3d, args.f, nr_cells=18, gap_width=5, viz=True)
        print(f"cell_centroids={cell_centroids}")
        print(f"patch w={patch_width} h={patch_height}")

    else:
        print(f"Method {method} not recognized")

