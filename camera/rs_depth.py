# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py

import cv2
import pyrealsense2 as rs
import numpy as np


def draw_rectangle(event, x, y, flags, param):
    global pt1, pt2, topLeft_clicked, botRight_clicked

    if event == cv2.EVENT_LBUTTONDOWN:

        if topLeft_clicked == True and botRight_clicked == True:
            topLeft_clicked = False
            botRight_clicked = False
            pt1 = (0, 0)
            pt2 = (0, 0)

        if topLeft_clicked == False:
            pt1 = (x, y)
            topLeft_clicked = True

        elif botRight_clicked == False:
            pt2 = (x, y)
            botRight_clicked = True


pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
botRight_clicked = False

try:
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    cv2.namedWindow('Depth')
    cv2.setMouseCallback('Depth', draw_rectangle)

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)

        cv2.putText(depth_image1, text="depth_scale={:1.5f}m".format(depth_scale), org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

        if topLeft_clicked:
            cv2.circle(depth_image1, center=pt1, radius=3, color=(0, 0, 255), thickness=-1)

        # draw rectangle
        if topLeft_clicked and botRight_clicked:
            cv2.rectangle(depth_image1, pt1, pt2, (0, 0, 255), 2)

            c00 = np.minimum(pt1[0], pt2[0])
            c01 = np.maximum(pt1[0], pt2[0])
            c10 = np.minimum(pt1[1], pt2[1])
            c11 = np.maximum(pt1[1], pt2[1])

            patch = depth_image[c10: c11, c00: c01]  # .flatten()

            aa = np.argwhere(patch > 0)

            if len(aa) > 0:
                aa[:, 0] = aa[:, 0] + c10
                aa[:, 1] = aa[:, 1] + c00

                d_avg = 0.0
                for i in range(len(aa)):
                    d_avg += depth_frame.get_distance(aa[i, 1],
                                                      aa[i, 0])  # depth_image[aa[i, 0], aa[i, 1]] * depth_scale
                d_avg /= len(aa)

                cv2.putText(depth_image1, text="d={:1.2f}m".format(d_avg), org=(150, 470),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                            thickness=2, lineType=cv2.LINE_AA)

        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', depth_image1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    pipeline.stop()


except Exception as e:
    print("exception!!!")
    print(e)
    cv2.destroyAllWindows()
    pipeline.stop()
    pass
