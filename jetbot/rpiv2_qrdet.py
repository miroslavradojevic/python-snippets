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
import pyzbar.pyzbar as pyzbar
from enum import Enum

class QRCodeDetector(Enum):
    ZBAR = 1
    OPENCV = 2

angle = 0.0
angle_last = 0.0

speed_gain = 0.35  # min=0.0, max=1.0, step=0.01
steering_gain = 0.01  # min=0.0, max=1.0, step=0.01
steering_dgain = 0.01  # min=0.0, max=0.5, step=0.001
steering_bias = 0.0  # min=-0.3, max=0.3, step=0.01

# Find QR code using ZBar
def decode(im):
    decoded_obj = pyzbar.decode(im)
    for obj in decoded_obj:
        print("{} : {}\n".format(obj.type, obj.data))
    return decoded_obj

# Display barcode and QR code location
def draw_qr(im, bbox):
    if bbox is not None:
        n = len(bbox)
        for j in range(n):
            cv2.line(im, tuple(bbox[j]), tuple(bbox[(j + 1) % n]), (0, 0, 255), 3)

    return im


def bbox_center(bbox):
    # bbox = np.squeeze(bbox, axis=1)
    polygon_centroid = np.mean(bbox, axis=0)
    return polygon_centroid


def control(im, bbox_center=None):
    global angle, angle_last, speed_gain, steering_gain, steering_dgain, steering_bias
    # TODO add check that center is within the image

    control_output = dict()

    if bbox_center is not None:
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

        control_output["left_motor"] = left_motor
        control_output["right_motor"] = right_motor
        control_output["x"] = x
        control_output["y"] = y
        control_output["angle"] = angle
        control_output["steering"] = steering

    else:
        control_output["left_motor"] = 0.0
        control_output["right_motor"] = 0.0
        control_output["x"] = np.nan
        control_output["y"] = np.nan
        control_output["angle"] = np.nan
        control_output["steering"] = np.nan

    return control_output

def qr_detector_opencv(decoder, input_image):
    # Detect and decode the QR code using OpenCV
    data, polygon, _ = decoder.detectAndDecode(input_image)
    # str, numpy.ndarray (4, 1, 2) | "", None
    if len(data) > 0:
        polygon = np.squeeze(polygon, axis=1)
        return data, polygon
    else:
        return "", None

def qr_detector_zbar(input_image):
    # Detect and decode the QR code using ZBar
    decoded_objects = pyzbar.decode(input_image) # decode(input_image)
    if len(decoded_objects) > 0:
        data = str(decoded_objects[0].data, 'utf-8')
        polygon = np.asarray(decoded_objects[0].polygon)
        return data, polygon
    else:
        return "", None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record video .avi using Jetson Nano with RPi_v2 camera using OpenCV")
    parser.add_argument("-fps", type=int, help="Frames per second - camera capture", default=21)
    parser.add_argument("-scale", type=int, default=4, help="")
    parser.add_argument("-record", type=int, default=0, help="")
    parser.add_argument("-speed_gain", type=float, default=0.35, help="")
    parser.add_argument("-steering_gain", type=float, default=0.05, help="")
    parser.add_argument("-method", type=int, default=1, help="")
    args = parser.parse_args()

    if args.scale < 1 or args.scale > 10:
        exit("Scale is out of accepted range")

    speed_gain = args.speed_gain

    try:
        QRCodeDetector(args.method)
    except:
        exit("Wrong method index. Exiting...")

    # Output directory used when saving frames individualy
    if args.record == 1:
        dir_path = join(expanduser("~"), "frame_" + time.strftime("%Y%m%d-%H%M%S"))  # basename(__file__)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    video_record = False  # Record .avi using frames captured in cam.value
    cam = Camera(Camera.RPI_V2_WIDTH // args.scale, Camera.RPI_V2_HEIGHT // args.scale, args.fps, video_record)

    robot = Robot()

    qrDecoder = cv2.QRCodeDetector()

    t0 = None
    t1 = None
    dt = None

    try:
        while True:
            polygon_qr_code = None
            data_qr_code = ""

            if args.record == 1:
                im_rec = np.copy(cam.value)

            if QRCodeDetector(args.method) == QRCodeDetector.ZBAR:
                data_qr_code, polygon_qr_code = qr_detector_zbar(cam.value)
            elif QRCodeDetector(args.method) == QRCodeDetector.OPENCV:
                data_qr_code, polygon_qr_code = qr_detector_opencv(qrDecoder, cam.value)

            found_qr_code = len(data_qr_code) > 0 and polygon_qr_code is not None

            if t0 is None:
                t0 = int(time.time() * 1000.0)
            else:
                t1 = int(time.time() * 1000.0)
                dt = t1 - t0  # milliseconds
                t0 = t1

            if dt is not None:
                print("FPS={:.1f}".format(1.0 / (dt / 1000.0)))

            if found_qr_code:
                bbox_xy = bbox_center(polygon_qr_code)
                print("QR Code ({}) center : {}\n".format(QRCodeDetector(args.method), bbox_xy), flush=True)
                # move robot
                ctrl = control(cam.value, bbox_xy)
                print("{}\n".format(ctrl), flush=True)

            else:
                print("QR Code ({}) not detected\n".format(QRCodeDetector(args.method)), flush=True)
                # Stop robot
                ctrl = control(cam.value, None)

            robot.left_motor.value = ctrl["left_motor"]
            robot.right_motor.value = ctrl["right_motor"]

            # Printout
            if args.record == 1:
                if found_qr_code:
                    im_rec = draw_qr(im_rec, polygon_qr_code)
                    # bbox_xy = np.expand_dims(bbox_xy, 0) # (2, ) => (1, 2)
                    bbox_xy = np.round(bbox_xy).astype(np.int32)  # .tolist()
                    cv2.circle(im_rec, (bbox_xy[0], bbox_xy[1]), 3, (0, 0, 255), 3)

                cv2.putText(im_rec, text="L,R={:+.2f},{:+.2f}".format(ctrl["left_motor"], ctrl["right_motor"]),
                            org=(10, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)
                cv2.putText(im_rec, text="steering={:+.2f}".format(ctrl["steering"]),
                            org=(10, 70),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)
                cv2.putText(im_rec, text="x,y={:+.2f},{:+.2f}".format(ctrl["x"], ctrl["y"]),
                            org=(10, 100),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)
                cv2.putText(im_rec, text="angle={:+.2f}".format(ctrl["angle"]),
                            org=(10, 130),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA)

                img_path = os.path.join(dir_path, 'cam_' + str(int(time.time() * 1000.0)) + '.jpg')
                cv2.imwrite(img_path, im_rec)

                print("{} {}\n".format(datetime.now(), img_path), flush=True)

            time.sleep(1e-4)

    except KeyboardInterrupt:
        cam.stop()
        cv2.destroyAllWindows()
