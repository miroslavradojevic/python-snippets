im_path = 'screen-test-rgb.jpg'

import cv2
# pip install ?
im_cv = cv2.imread(im_path)
im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
print(type(im_cv), im_cv.shape)

from matplotlib.image import imread
# 
im_matplotlib = imread(im_path)
print(type(im_matplotlib), im_matplotlib.shape)

import skimage
print(skimage.__version__)

from skimage import io
# https://scikit-image.org/docs/stable/install.html
# pip install scikit-image
im_skimage = io.imread(im_path)

print(type(im_skimage), im_skimage.shape)

# if not os.path.isfile(image_path):
#     print(image_path, " is not a file.")
#     quit()

# if not image_path.endswith(".tif"):
#     print(image_path, " needs to be tif (stack).")

# imageio
# from scipy.misc import imread