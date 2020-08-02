#!/usr/bin/env python
import cv2

# Connect to computer default camera
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame dimensions, w=%d, h=%d" % (width, height))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(type(gray), gray.shape, gray[0].dtype)

cap.release()
cv2.destroyAllWindows()