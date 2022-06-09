import cv2 
from math import inf
import numpy as np
from numpy import uint8
import pandas as pd
from os.path import exists, basename, join

def extract_patches(img, cell_centroids, patch_width=64, patch_height=128, annot=None, prefix="patch"):
    if img.ndim!=3:
        print("Input image needs to be 3D image stack")
        return
    
    if img.dtype != uint8:
        pass

    out_dir = f"patch_{patch_width}x{patch_height}"
    train_cat = ["train", "validation"]
    class_cat = ["adhesivelayer", "batterycell"]
    for tc in train_cat:
        for cc in class_cat:
            d= join(out_dir, tc, cc)
            if not exists(d):
                makedirs(d)

    for layer in range(img.shape[0]):
        cell_cnt = 1
        for cc in cell_centroids:
            yp = int(round(cc[1]))
            xp = int(round(cc[0]))
            patch = img[layer, yp-patch_height//2:yp+patch_height//2, xp-patch_width//2:xp+patch_width//2]
            patch = ((patch / patch.max()) * 255).astype(np.uint8) # normalizes data in range 0 - 255
            
            # use cell_cnt and annot to say whether it is adhesive layer or not
            layer_rng = np.squeeze(annot[np.where(annot[:,0]==cell_cnt), 1:])

            is_adhesive = False
            for row in layer_rng:
                if row[0] <= (layer+1) <= row[1]:
                    is_adhesive = True
                    break

            is_train = layer < img.shape[0]//2

            out_path = join(out_dir, train_cat[0 if is_train else 1], class_cat[0 if is_adhesive else 1], f"{prefix}_layer{layer+1}_cell{cell_cnt}.png")
            print(out_path)

            cv2.imwrite(out_path, patch)
            
            cell_cnt += 1

def viz_battery_division(img, x1, xvert):
    # visualize vertical divisions
    img_viz = np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(uint8)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    ih,iw = img.shape
    for xi in xvert:
        img_viz = cv2.line(img_viz, (x1+xi,0), (x1+xi,ih), (0, 255, 255), 2)
        img_viz = cv2.line(img_viz, (x1+xi,0), (x1+xi,ih), (0, 255, 255), 2)
    cv2.imwrite("viz_battery_division.jpg", img_viz) 

def viz_cell_centroids(img, ccs):
    img_viz = np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(uint8)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    for cc in ccs:
        img_viz = cv2.circle(img_viz, tuple(map(int, cc)), 8, (0, 0, 255), -1)
    cv2.imwrite("viz_cell_centroids.jpg", img_viz)



def viz_battery_detection(img, x1, x2, y1, y2):
    # visualize battery boundaries
    img_viz = np.round((img - img.min()) / (img.max() - img.min()) * 255).astype(uint8)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    ih,iw = img.shape
    img_viz = cv2.line(img_viz, (x1,0), (x1,ih), (0, 255, 255), 2)
    img_viz = cv2.line(img_viz, (x2,0), (x2,ih), (0, 255, 255), 2)
    img_viz = cv2.line(img_viz, (0,y1), (iw,y1), (0, 255, 255), 2)
    img_viz = cv2.line(img_viz, (0,y2), (iw,y2), (0, 255, 255), 2)
    cv2.imwrite("viz_battery_detection.jpg", img_viz) 

def detect_cells(img, gap_width=7, nr_cells=18):
    if (gap_width % 2) == 0 or (nr_cells % 2) == 1:
        print(f"Gap width ({gap_width}) must be odd number and number of cells ({nr_cells}) must be even number")
        return None

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

    x_out.sort()
    return x_out



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
    


def extract_cells(img, img3d, img_path, nr_cells=18, gap_width=5, viz=False):
        x1,x2,y1,y2 = detect_battery_rectangle(img)
        
        if viz:
            viz_battery_detection(img, x1, x2, y1, y2)

        img_crop = img[y1:y2, x1:x2]
        
        x_vert = detect_cells(img_crop, gap_width, nr_cells)

        if x_vert is None:
            return None, None, None

        if viz:
            viz_battery_division(img, x1, x_vert)

        # extract cell centroids
        cell_centroids = []
        cell_widths = []
        for i in range(1, len(x_vert)):
            x_r = x_vert[i]
            x_l = x_vert[i-1]
            cell_centroids.append((x1+0.5*(x_l+x_r), 0.5*(y1+y2)))
            cell_widths.append(x_r - x_l)

        if viz:
            viz_cell_centroids(img, cell_centroids)
        
        patch_width = int(round(np.mean(cell_widths)))
        patch_height = int(round(y2 - y1))
        
        annot_path = img_path.replace(".raw", ".csv")
        if exists(annot_path):
            annot = pd.read_csv(annot_path, sep=',', header=0).to_numpy()
            extract_patches(img3d, cell_centroids, patch_width, patch_height, annot, basename(img_path.replace(".raw", "")))

        return cell_centroids, patch_width, patch_height

