# pip install scikit-image
import skimage.io as io 
from skimage.util import compare_images
import numpy as np

im0 = io.imread('a0.jpg') # median of source images
im1 = io.imread('a1.jpg') # source image 1
im2 = io.imread('a2.jpg') # source image 2
im3 = io.imread('a3.jpg') # source image 3

im_all = np.copy(im0)
th = 0.1

# d = np.max(np.abs(im2 - im0), -1)
d = compare_images(im1, im0, method='diff')
d= np.max(np.abs(d), -1)
im_all[d>th] = im1[d>th]
io.imsave("d1.jpg", d>th)

d = compare_images(im2, im0, method='diff')
d= np.max(np.abs(d), -1)
im_all[d>th] = im2[d>th]
io.imsave("d2.jpg", d>th)

d = compare_images(im3, im0, method='diff')
d= np.max(np.abs(d), -1)
im_all[d>th] = im3[d>th]
io.imsave("d3.jpg", d>th)

io.imsave("im_all.jpg", im_all)

# print("{}, {}".format(type(im), im.shape))
# plt.imshow(diff_12)
# plt.show()
# blend_rotated = compare_images(img1, img2, method='blend')