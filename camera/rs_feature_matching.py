import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

SAVE_RECORDING = True
W = 640
H = 480

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# MACOS AND LINUX: *'XVID' (MacOS users may want to try VIDX as well)
# WINDOWS *'VIDX'
if SAVE_RECORDING:
    writer = cv2.VideoWriter("sift-fe_" + time.strftime("%Y%m%d-%H%M%S") + ".mp4", cv2.VideoWriter_fourcc(*'VIDX'), 30, (2*W, H))

pipeline.start(config)

image1 = None
depth1 = None
kp1 = None
des1 = None

sift = cv2.xfeatures2d.SIFT_create()
# orb = cv2.ORB_create()
bf = cv2.BFMatcher()

d2_d1_ratio = 0.5

v = None

try:
    while True:
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth2 = depth_frame # np.asanyarray(depth_frame.get_data())
        image2 = np.asanyarray(color_frame.get_data())

        kp2, des2 = sift.detectAndCompute(image2, None)

        if image1 is not None:
            matches = bf.knnMatch(des1, des2, k=2)

            good = []
            for match1, match2 in matches:
                if match1.distance < d2_d1_ratio * match2.distance:
                    good.append([match1])

            if len(good) > 0:
                v = np.zeros(2)
                d = 0
                for i in range(len(good)):
                    # print(type(good[0]), " ", len(good[0]))
                    p1 = kp1[good[0][0].queryIdx].pt
                    d1 = depth1.get_distance(int(p1[0]), int(p1[1]))
                    p2 = kp2[good[0][0].trainIdx].pt
                    d2 = depth2.get_distance(int(p2[0]), int(p2[1]))

                    v = v + np.subtract(p2, p1)
                    d = d + (d2-d1)

                v = v / len(good)
                d = d /len(good)
                # print(v)


            sift_matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good, None, flags=2)

            # cv2.imshow("Images", np.hstack((image1, image2)))

            cv2.putText(sift_matches, text='Press ESC to exit', org=(0, int(0.99*sift_matches.shape[0])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255), thickness=1,
                        lineType=cv2.LINE_AA)


            cv2.putText(sift_matches, text='d2/d1='+str(d2_d1_ratio),
                        org=(int(0.45*sift_matches.shape[1]), int(0.99*sift_matches.shape[0])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255), thickness=1,
                        lineType=cv2.LINE_AA)

            if v is not None:
                cv2.putText(sift_matches, text="vx={:+1.2f}, vy={:+1.2f}, vz={:+1.2f}".format(v[0], v[1], d),
                        org=(int(0.80*sift_matches.shape[1]), int(0.99*sift_matches.shape[0])),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255), thickness=1,
                        lineType=cv2.LINE_AA)

            cv2.namedWindow('SIFT', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("SIFT", sift_matches)
            if SAVE_RECORDING:
                writer.write(sift_matches)

        # Recursion
        image1 = image2
        kp1 = kp2
        des1 = des2
        depth1 = depth2

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    if SAVE_RECORDING:
        writer.release()