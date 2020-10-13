#!/usr/bin/env python
import argparse
import os
import time
from datetime import datetime
from os.path import expanduser, join
import sys
import cv2
import numpy as np
from rpiv2 import Camera
from jetbot import Robot

angle = 0.0
angle_last = 0.0

speed_gain = 0.4  # min=0.0, max=1.0, step=0.01
steering_gain = 0.01  # min=0.0, max=1.0, step=0.01
steering_dgain = 0.01  # min=0.0, max=0.5, step=0.001
steering_bias = 0.0  # min=-0.3, max=0.3, step=0.01


# Display barcode and QR code location
def draw_qr(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[(j + 1) % n][0]), (0, 0, 255), 3)

    return im


def bbox_center(bbox):
    bbox = np.squeeze(bbox, axis=1)
    return np.mean(bbox, axis=0)


def control(im, bbox_center):
    global angle, angle_last, speed_gain, steering_gain, steering_dgain, steering_bias
    x_center = im.shape[0] / 2.0
    # y_center = im.shape[1] / 2.0

    x_bbox = bbox_center[0]
    y_bbox = bbox_center[1]

    x = (x_bbox - x_center) / x_center  # -1 : 1
    y = (y_bbox - im.shape[1]) / im.shape[1]  # 0 : 1

    angle = np.arctan2(x, y)  # angles in radians, in the range [-pi, pi].
    pid = angle * steering_gain + (angle - angle_last) * steering_dgain
    angle_last = angle

    steering = pid + steering_bias

    left_motor = max(min(speed_gain + steering, 1.0), 0.0)
    right_motor = max(min(speed_gain - steering, 1.0), 0.0)

    control_output = dict()
    control_output["left_motor"] = left_motor
    control_output["right_motor"] = right_motor
    control_output["x"] = x
    control_output["y"] = y
    control_output["angle"] = angle
    control_output["steering"] = steering

    return control_output  # left_motor, right_motor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record video .avi using Jetson Nano with RPi_v2 camera using OpenCV")
    parser.add_argument("-fps", type=int, help="Frames per second - camera capture", default=21)
    parser.add_argument("-scale", type=int, default=4, help="")
    parser.add_argument("-record", type=int, default=0, help="")
    parser.add_argument("-speed_gain", type=float, default=0.35, help="")
    parser.add_argument("-steering_gain", type=float, default=0.05, help="")
    args = parser.parse_args()

    if args.scale < 1 or args.scale > 10:
        exit("Scale is out of accepted range")

    # Output directory used when saving frames individualy
    dir_path = join(expanduser("~"), "framerec_" + time.strftime("%Y%m%d-%H%M%S"))  # basename(__file__)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    video_record = False  # Need only frame in cam.value
    cam = Camera(Camera.RPI_V2_WIDTH // args.scale, Camera.RPI_V2_HEIGHT // args.scale, args.fps, video_record)

    robot = Robot()

    qrDecoder = cv2.QRCodeDetector()

    try:
        while True:
            # Detect and decode the qrcode
            data, bbox, rectifiedImage = qrDecoder.detectAndDecode(cam.value)

            img_path = os.path.join(dir_path, 'cam_' + str(int(time.time() * 1000.0)) + '.jpg')

            if len(data) > 0:
                bbox_xy = bbox_center(bbox)

                sys.stdout.write("QR Code center : {}\n".format(bbox_xy))
                sys.stdout.flush()

                ctrl = control(cam.value, bbox_xy)

                sys.stdout.write(str(ctrl))
                sys.stdout.flush()

                robot.left_motor.value = ctrl["left_motor"]
                robot.right_motor.value = ctrl["right_motor"]

                if args.record == 1:
                    im_rec = np.copy(cam.value)
                    im_rec = draw_qr(im_rec, bbox)


                    control_frame = cv2.putText(im_rec, text="L,R={:+.2f},{:+.2f}".format(ctrl["left_motor"], ctrl["right_motor"]), org=(10, 40),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    control_frame = cv2.putText(im_rec, text="steering={:+.2f}".format(ctrl["steering"]), org=(10, 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    control_frame = cv2.putText(im_rec, text="x,y={:+.2f},{:+.2f}".format(ctrl["x"], ctrl["x"]), org=(10, 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    control_frame = cv2.putText(im_rec, text="angle={:+.2f}".format(ctrl["angle"]), org=(10, 130),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                    # cv2.circle(img, center=pt1, radius=3, color=(0, 0, 255), thickness=-1)
                    cv2.circle(control_frame, (int(bbox_xy[0]), int(bbox_xy[1])), 3, (0, 0, 255), 3)
                    cv2.imwrite(img_path, control_frame)
                    sys.stdout.write("{} {}\n".format(datetime.now(), img_path))
                    sys.stdout.flush()
            else:
                # stop robot
                robot.left_motor.value = 0.0
                robot.right_motor.value = 0.0

                sys.stdout.write("QR Code not detected\n")
                sys.stdout.flush()

                if args.record == 1:
                    im_rec = np.copy(cam.value) # int(0.95 * cam.value.shape[1])
                    cv2.putText(im_rec, text="L,R={:+.2f},{:+.2f}".format(np.nan, np.nan), org=(10, 40),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.putText(im_rec, text="steering={:+.2f}".format(np.nan), org=(10, 70),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.putText(im_rec, text="x,y={:+.2f},{:+.2f}".format(np.nan, np.nan), org=(10, 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.putText(im_rec, text="angle={:+.2f}".format(np.nan), org=(10, 130),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 255, 255), thickness=1,
                                lineType=cv2.LINE_AA)
                    cv2.imwrite(img_path, im_rec)
                    sys.stdout.write("{} {}\n".format(datetime.now(), img_path))
                    sys.stdout.flush()
            time.sleep(1e-6)
    except KeyboardInterrupt:
        cam.stop()
        cv2.destroyAllWindows()
