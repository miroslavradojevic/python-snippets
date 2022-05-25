#!/usr/bin/env python3
import sys
import numpy as np
import argparse
import SimpleITK as sitk
import cv2
from numpy import uint8
from read.image import string_to_pixelType, my_func, read_raw
from os.path import exists, splitext, join
from os import makedirs
from time import strftime
from math import inf
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import matplotlib.pyplot as plt

def raw_to_npy(raw_path, raw_size, raw_big_endian, raw_type, raw_min = None, raw_max = None):
    # Read the image
    image = read_raw(binary_file_name=raw_path,
        image_size=raw_size,
        sitk_pixel_type=string_to_pixelType[raw_type],
        big_endian=raw_big_endian)

    image_npy = sitk.GetArrayFromImage(image)

    if raw_min is not None:
        image_npy[image_npy<raw_min] = raw_min
    
    if raw_max is not None:
        image_npy[image_npy>raw_max] = raw_max

    return image_npy

def raw_to_tif(raw_path, raw_size, raw_big_endian, raw_type, raw_min = None, raw_max = None):
    # Read the image
    image = read_raw(binary_file_name=raw_path,
        image_size=raw_size,
        sitk_pixel_type=string_to_pixelType[raw_type],
        big_endian=raw_big_endian)
    
    # print(f"{type(image)} H={image.GetHeight()}, W={image.GetWidth()}, D={image.GetDepth()}")

    # crop
    th = sitk.ThresholdImageFilter()
    if raw_min is not None:
        th.SetLower(raw_min)
    if raw_max is not None:
        th.SetUpper(raw_max)
    if raw_min is not None or raw_max is not None:
        image = th.Execute(image)

    # save image as tif stack
    out_file_name = splitext(args.f)[0] + f"_converted_crop_{raw_min}_{raw_max}.tif"
    sitk.WriteImage(image, out_file_name)
    print(f"Exported to\t{out_file_name}")

def detect_battery_rectangle(img):
    img_integral = cv2.integral(img)
    img_height, img_width = img.shape
    
    w_min = int(round(img_width * 0.10))
    w_max = int(round(img_width * 0.95))

    h_min = int(round(img_height * 0.10))
    h_max = int(round(img_height * 0.95))

    x1_min = int(round(img_width * 0.05))
    x1_max = int(round(img_width * 0.5))
    x2_min = int(round(img_width * 0.5))
    x2_max = int(round(img_width * 0.95))

    y1_min = int(round(img_height * 0.05))
    y1_max = int(round(img_height * 0.5))
    y2_min = int(round(img_height * 0.5))
    y2_max = int(round(img_height * 0.95))

    step = 1
    score_max = -inf
    x1_ = None
    x2_ = None
    
    # loop through the sampled rectangles
    for x1 in range(x1_min, x2_max, step):
        for x2 in range(x2_min, x2_max, step):
            if x2>=x1 and w_min <= x2-x1 <= w_max:
                # sum = bottom_right + top_left - top_right - bottom_left
                s1 = (img_integral[-1,x1] + img_integral[0,0] - img_integral[0,x1] - img_integral[-1,0])/float(x1*img_height)
                s2 = (img_integral[-1,x2] + img_integral[0,x1] - img_integral[0,x2] - img_integral[-1,x1])/float((x2-x1)*img_height)
                s3 = (img_integral[-1,-1] + img_integral[0,x2] - img_integral[0,-1] - img_integral[-1,x2])/float((img_width-x2)*img_height)
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
                s1 = (img_integral[y1,-1] + img_integral[0,0] - img_integral[y1,0] - img_integral[0,-1])/float(y1*img_width)
                s2 = (img_integral[y2,-1] + img_integral[y1,0] - img_integral[y2,0] - img_integral[y1,-1])/float((y2-y1)*img_width)
                s3 = (img_integral[-1,-1] + img_integral[y2,0] - img_integral[-1,0] - img_integral[y2,-1])/float((img_height-y2)*img_width)
                score = s2 - 0.5 * (s1 + s3)
                if score > score_max:
                    score_max = score
                    y1_ = y1
                    y2_ = y2

    return x1_,x2_,y1_,y2_
    
def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()

def detect_cells(img, gap_width=7, nr_cells=18):
    # TODO check nr_cells  must be even numbers gap_width must be odd

    img_integral = cv2.integral(img)
    img_height, img_width = img.shape
    step = 1

    x_out = []

    nhood = int(round(img_width/nr_cells)*0.25)

    # central division
    score_optim = -inf
    x_optim = None
    for dx in range(-nhood, nhood, step):
        x0 = img_width//2 + dx 
        x1 = x0 - gap_width//2
        x2 = x0 + gap_width//2
        x11 = x1 - gap_width
        x22 = x2 + gap_width
        s0 = (img_integral[-1,x2]  + img_integral[0,x1]  - img_integral[0,x2]  - img_integral[-1,x1]) /float(gap_width*img_height)
        s1 = (img_integral[-1,x1]  + img_integral[0,x11] - img_integral[0,x1]  - img_integral[-1,x11])/float(gap_width*img_height)
        s2 = (img_integral[-1,x22] + img_integral[0,x2]  - img_integral[0,x22] - img_integral[-1,x2]) /float(gap_width*img_height)
        score = 0.5*(s1+s2)-s0
        if score > score_optim:
            score_optim = score
            x_optim = x0
            
    x_out.append(x_optim)

    x_left = x_right = x_optim

    # expand from center towards right
    for i in range(0, nr_cells//2): 
        score_optim = -inf
        x_optim = None
        for dx in range(-nhood, nhood, step):
            x0 = x_right + int(round(img_width/nr_cells)) + dx
            x1 = x0 - gap_width//2
            x2 = x0 + gap_width//2
            x11 = x1 - gap_width
            x22 = x2 + gap_width

            if x22 <= img_width:
                s0 = (img_integral[-1,x2]  + img_integral[0,x1]  - img_integral[0,x2]  - img_integral[-1,x1]) /float(gap_width*img_height)
                s1 = (img_integral[-1,x1]  + img_integral[0,x11] - img_integral[0,x1]  - img_integral[-1,x11])/float(gap_width*img_height)
                s2 = (img_integral[-1,x22] + img_integral[0,x2]  - img_integral[0,x22] - img_integral[-1,x2]) /float(gap_width*img_height)
                score = 0.5*(s1+s2)-s0
                if score > score_optim:
                    score_optim = score
                    x_optim = x0
        
        if x_optim is not None:
            x_out.append(x_optim)
            x_right = x_optim
        else:
            break # stop further with for-loop first time score was not found

    # expand from center towards left
    for i in range(0, nr_cells//2):
        score_optim = -inf
        x_optim = None
        for dx in range(-nhood, nhood, step):
            x0 = x_left - int(round(img_width/nr_cells)) + dx
            x1 = x0 - gap_width//2
            x2 = x0 + gap_width//2
            x11 = x1 - gap_width
            x22 = x2 + gap_width

            if x11 >= 0:
                s0 = (img_integral[-1,x2]  + img_integral[0,x1]  - img_integral[0,x2]  - img_integral[-1,x1]) /float(gap_width*img_height)
                s1 = (img_integral[-1,x1]  + img_integral[0,x11] - img_integral[0,x1]  - img_integral[-1,x11])/float(gap_width*img_height)
                s2 = (img_integral[-1,x22] + img_integral[0,x2]  - img_integral[0,x22] - img_integral[-1,x2]) /float(gap_width*img_height)
                score = 0.5*(s1+s2)-s0
                if score > score_optim:
                    score_optim = score
                    x_optim = x0
        
        if x_optim is not None:
            x_out.append(x_optim)
            x_left = x_optim
        else:
            break # stop further with for-loop first time score was not found

    print(type(x_out), x_out)
    x_out.sort()
    return x_out

    x_ = [None] *  (nr_cells+1)

    div_min=int(round(img_width/nr_cells*0.8))
    div_max=int(round(img_width/nr_cells*1.2))
    div_boundary = int(round(img_width/nr_cells*0.5))

    for i in range(0, nr_cells+1):
        score_max = -inf
        if i==0:
            for w in range(0, div_boundary, step):
                x1 = w #+ 0
                if x1 <= img_width:
                    x2 = x1 + gap_width
                    s1 = (img_integral[-1,x1] + img_integral[0,0] - img_integral[0,x1] - img_integral[-1,0])/float(x1*img_height)
                    s2 = (img_integral[-1,x2] + img_integral[0,x1] - img_integral[0,x2] - img_integral[-1,x1])/float((x2-x1)*img_height)
                    s3 = (img_integral[-1,-1] + img_integral[0,x2] - img_integral[0,-1] - img_integral[-1,x2])/float((img_width-x2)*img_height)
                    score = 0.5 * (s1 + s3) - s2
                    if score > score_max:
                        score_max = score
                        x_[i] = x1
                else:
                    break

        else:
            for w in range(div_min, div_max, step):
                x1 = w + (x_[i-1]+gap_width) #((x_[i-1]+gap_width) if i!=0 else 0)
                if x1 <= img_width:
                    x2 = x1 + gap_width
                    s1 = (img_integral[-1,x1] + img_integral[0,0] - img_integral[0,x1] - img_integral[-1,0])/float(x1*img_height)
                    s2 = (img_integral[-1,x2] + img_integral[0,x1] - img_integral[0,x2] - img_integral[-1,x1])/float((x2-x1)*img_height)
                    s3 = (img_integral[-1,-1] + img_integral[0,x2] - img_integral[0,-1] - img_integral[-1,x2])/float((img_width-x2)*img_height)
                    score = 0.5 * (s1 + s3) - s2
                    if score > score_max:
                        score_max = score
                        x_[i] = x1
                else:
                    break

    return x_

def extract_patches(img, cell_centroids, patch_width=64, patch_height=64):
    if img.ndim!=3:
        print("Input image needs to be 3D image stack")
        return
    
    if img.dtype != uint8:
        pass

    out_dir = f"cell_patches_{patch_width}x{patch_height}"
    if not exists(out_dir):
        makedirs(out_dir) 

    for layer in range(img.shape[0]):
        cell_cnt = 1
        for cc in cell_centroids:
            yp = int(round(cc[1]))
            xp = int(round(cc[0]))
            patch = img[layer, yp-patch_height//2:yp+patch_height//2, xp-patch_width//2:xp+patch_width//2]
            cv2.imwrite(join(out_dir, f"patch_layer{layer+1}_cell{cell_cnt}.tif"), patch)
            cell_cnt += 1

def viz_battery_detection(img, x1, x2, y1, y2):
    # visualize battery boundaries
    print(x1, x2, y1, y2)
    img_viz = np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(uint8)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    ih,iw = img.shape
    img_viz = cv2.line(img_viz, (x1,0), (x1,ih), (0, 255, 255), 2)
    img_viz = cv2.line(img_viz, (x2,0), (x2,ih), (0, 255, 255), 2)
    img_viz = cv2.line(img_viz, (0,y1), (iw,y1), (0, 255, 255), 2)
    img_viz = cv2.line(img_viz, (0,y2), (iw,y2), (0, 255, 255), 2)
    cv2.imwrite("viz_battery_detection.jpg", img_viz) 

def viz_battery_division(img, x1, xvert):
    # visualize vertical divisions
    img_viz = np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(uint8)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    ih,iw = img.shape
    for xi in xvert:
        img_viz = cv2.line(img_viz, (x1+xi,0), (x1+xi,ih), (0, 255, 255), 2)
        img_viz = cv2.line(img_viz, (x1+xi,0), (x1+xi,ih), (0, 255, 255), 2)
    cv2.imwrite("viz_battery_division.jpg", img_viz) 



def extract_cells(img, nr_cells=18, gap_width=5, out_dir="patches", annot=None, viz=False):
        x1,x2,y1,y2 = detect_battery_rectangle(img)
        
        if viz:
            viz_battery_detection(img, x1, x2, y1, y2)

        img_crop = img[y1:y2, x1:x2]
        
        x_vert = detect_cells(img_crop, gap_width, nr_cells)

        if x_vert is None:
            return 

        print(f"x_vert={x_vert} | {len(x_vert)}")

        viz_battery_division(img, x1, x_vert)
        
        if True:
            return

        # extract cell centroids
        cell_centroids = []
        for i in range(len(x_vert)):
            x_right = x_vert[i]
            x_left = x_vert[i-1] if i!=0 else 0
            cell_centroids.append((x1+0.5*(x_left+x_right), 0.5*(y1+y2)))
        cell_centroids.append((x1+0.5*((x2-x1)+x_right), 0.5*(y1+y2)))

        print(f"cell_centroids=\n{cell_centroids}")

        # visualize cells
        for cc in cell_centroids:
            img_viz = cv2.circle(img_viz, tuple(map(int, cc)), 8, (0, 255, 0), -1)
        cv2.imwrite("cell_centroids.jpg", img_viz)

        patch_width = 100
        patch_height = y2 - y1
        extract_patches(img3d, cell_centroids, patch_width, patch_height)

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
        img = raw_to_npy(args.f, args.sz, args.big_endian, args.type)
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
        print(img3d.shape, type(img3d), img3d.dtype)
        img = np.mean(img3d, 0) # z-projection, median or mean

        extract_cells(img, nr_cells=18, gap_width=5, out_dir="patches", annot=None, viz=True)

    else:
        print(f"Method {method} not recognized")

