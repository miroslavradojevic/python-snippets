import cv2
im_cv = cv2.imread('/home/miro/Downloads/frame_BGR.jpg')
im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
print(type(im_cv), im_cv.shape)