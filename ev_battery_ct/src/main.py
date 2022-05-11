#!/usr/bin/env python3
import argparse
import SimpleITK as sitk
import cv2
from numpy import uint8
from read.image import string_to_pixelType, my_func, read_raw
from os.path import exists, splitext
from math import inf

print(sitk.__version__)

def raw_to_tif(file, size, big_endian, type):
    # Read the image
    image = read_raw(binary_file_name=args.f,
        image_size=args.sz,
        sitk_pixel_type=string_to_pixelType[args.type],
        big_endian=args.big_endian)
    
    print(f"H={image.GetHeight()}, W={image.GetWidth()}, D={image.GetDepth()}")

    # save image
    out_file_name = splitext(args.f)[0] + "_converted.tif"
    sitk.WriteImage(image, out_file_name)
    print(f"Exported to\t{out_file_name}")

def raw_to_tif_cropped(file, size, big_endian, type, min_val, max_val):
    # Read the image
    image = read_raw(binary_file_name=args.f,
        image_size=args.sz,
        sitk_pixel_type=string_to_pixelType[args.type],
        big_endian=args.big_endian)
    
    # crop
    th = sitk.ThresholdImageFilter()
    th.SetLower(min_val)
    th.SetUpper(max_val)
    image_out = th.Execute(image)

    # save image
    out_file_name = splitext(args.f)[0] + f"_cropped_{min_val}_{max_val}.tif"
    sitk.WriteImage(image_out, out_file_name)
    print(f"Exported to\t{out_file_name}")

def detect_battery_rectangle(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print(type(img), img.shape, img.dtype)
    if img.dtype == uint8:
        print("yes")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_integral = cv2.integral(img)
    print(type(img_integral), img_integral.shape, img_integral.dtype)
    
    img_height, img_width = img.shape
    
    w_min = int(round(img_width * 0.2))
    w_max = int(round(img_width * 0.8))

    h_min = int(round(img_height * 0.2))
    h_max = int(round(img_height * 0.8))

    x1_min = int(round(img_width * 0.05))
    x1_max = int(round(img_width * 0.5))
    x2_min = int(round(img_width * 0.5))
    x2_max = int(round(img_width * 0.95))
    print(x1_min, x1_max, x2_min, x2_max)

    y1_min = int(round(img_height * 0.1))
    y1_max = int(round(img_height * 0.5))
    y2_min = int(round(img_height * 0.5))
    y2_max = int(round(img_height * 0.9))
    print(y1_min, y1_max, y2_min, y2_max)

    step = 1
    score_max = -inf
    x1_ = None
    x2_ = None
    
    # loop through the sampled rectangles
    for x1 in range(x1_min, x2_max, step):
        for x2 in range(x2_min, x2_max, step):
            if x2>=x1 and w_min <= x2-x1 <= w_max:
                # compute s1,...,s9
                # sum = bottom_right + top_left - top_right - bottom_left
                s1 = img_integral[-1,x1] + img_integral[0,0] - img_integral[0,x1] - img_integral[-1,0]
                # s1 /= float(x1*img_height)
                
                s2 = img_integral[-1,x2] + img_integral[0,x1] - img_integral[0,x2] - img_integral[-1,x1]
                # s2 /= float((x2-x1)*img_height)

                s3 = img_integral[-1,-1] + img_integral[0,x2] - img_integral[0,-1] - img_integral[-1,x2]
                # s3 /= float((x2-x1)*img_height)

                score = s2 - 0.5 * (s1 + s3)
                if score > score_max:
                    score_max = score
                    x1_ = x1
                    x2_ = x2

    score_max = -inf
    y1_ = None
    y2_ = None
    for y1 in range(y1_min, y1_max, step):
        for y2 in range(y2_min, y2_max, step):
            if y2>=y1 and h_min <= y2-y1 <= h_max:
                s1 = img_integral[-1,x1] + img_integral[0,0] - img_integral[0,x1] - img_integral[-1,0]
                s2 = img_integral[-1,x2] + img_integral[0,x1] - img_integral[0,x2] - img_integral[-1,x1]
                s3 = img_integral[-1,-1] + img_integral[0,x2] - img_integral[0,-1] - img_integral[-1,x2]

                score = s2 - 0.5 * (s1 + s3)
                if score > score_max:
                    score_max = score
                    x1_ = x1
                    x2_ = x2

    img_viz = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    p1 = (x1_,0)
    p2 = (x1_,img_height)
    img_viz = cv2.line(img_viz, p1, p2, (0, 255, 255), 2)

    p1 = (x2_,0)
    p2 = (x2_,img_height)
    img_viz = cv2.line(img_viz, p1, p2, (0, 255, 255), 2)

    p1 = (0, y1_min)
    p2 = (img_width, y1_min)
    img_viz = cv2.line(img_viz, p1, p2, (0, 255, 255), 2)

    p1 = (0, y2_max)
    p2 = (img_width, y2_max)
    img_viz = cv2.line(img_viz, p1, p2, (0, 255, 255), 2)

    cv2.imwrite("test.jpg", img_viz) 
    

if __name__=='__main__':
    psr = argparse.ArgumentParser(description='EV battery image analysis')
    psr.add_argument('-m', type=str, required=True, help='Select method')
    psr.add_argument('-f', type=str, required=False, help='Path to the input image file (image stack, binary raw)')
    psr.add_argument('-sz', required=False, nargs='+', help="(width,height,length)", type=int)
    psr.add_argument("-big_endian", required=False, type=lambda v: v.lower() in {"1", "true"}, default=False, help="\'false\' for little-endian or \'true\' for big-endian")
    psr.add_argument('-type', required=False, default="sitkFloat32", help="SimpleITK pixel type (default: sitkFloat32)")
    # add min, max for cropping

    args = psr.parse_args()

    method = args.m.upper()

    if method == "RAW_TO_TIF":
        if not any(d is None for d in (args.f, args.sz, args.big_endian, args.type)):
            if exists(args.f):
                if splitext(args.f)[1].upper() ==  ".RAW":
                    # read input .raw image and save as .tif
                    raw_to_tif(args.f, args.sz, args.big_endian, args.type)
                else:
                    print(f"File extension must be .raw")
            else:
                print(f"File {args.f} could not be found")
        else:
            print("Parameters are missing")
    
    elif method == "RAW_TO_TIF_CROPPED":
        if not any(d is None for d in (args.f, args.sz, args.big_endian, args.type)):
            if exists(args.f):
                if splitext(args.f)[1].upper() ==  ".RAW":
                    # read input .raw image, crop between [min_val, max_val] and save as .tif
                    raw_to_tif_cropped(args.f, args.sz, args.big_endian, args.type, 0.00, 0.05)
                else:
                    print(f"File extension must be .raw")
            else:
                print(f"File {args.f} could not be found")
        else:
            print("Parameters are missing")        
        
    elif method == "RAW_TO_NUMPY":
        # sitk.GetArrayFromImage
        # pass
        if not any(d is None for d in (args.f, args.sz, args.big_endian, args.type)):
            # raw_to_numpy(args.f, args.sz, args.big_endian, args.type)
            pass
        else:
            print("Parameters are missing")
    elif method == "TIF_TO_NUMPY":
        pass
    elif method == "DETECT_BATTERY_RECTANGLE":
        if args.f is not None:
            if exists(args.f):
                if splitext(args.f)[1].upper() ==  ".TIF":
                    detect_battery_rectangle(args.f)
                else:
                    print(f"File extension must be .tif")
            else:
                print(f"File {args.f} could not be found")
        else:
            print("Input image is missing")
    elif method == "TEST":
        print("Test")
    else:
        print(f"Method {method} not recognized")

