#!/usr/bin/env python

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

pt1 = (0, 0)
pt2 = (0, 0)
topLeft_clicked = False
botRight_clicked = False

depth_scale = 0.001

class ReadImage(object):
    def __init__(self):
        cv2.namedWindow('Read image')
        cv2.setMouseCallback('Read image', self.draw_rectangle)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.read_image_callback)


    def read_image_callback(self, img):
        rospy.loginfo("Image (w={0}, h={1}) received".format(img.width, img.height))
        try:
            # http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
            frame = self.bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
            frame1 = cv2.applyColorMap(cv2.convertScaleAbs(frame, alpha=0.05), cv2.COLORMAP_JET)

            if topLeft_clicked:
                cv2.circle(frame1, center=pt1, radius=3, color=(0, 0, 255), thickness=-1)

            if topLeft_clicked and botRight_clicked:
                cv2.rectangle(frame1, pt1, pt2, (0, 0, 255), 2)

                c00 = np.minimum(pt1[0], pt2[0])
                c01 = np.maximum(pt1[0], pt2[0])
                c10 = np.minimum(pt1[1], pt2[1])
                c11 = np.maximum(pt1[1], pt2[1])

                patch = frame[c10: c11, c00: c01]  # .flatten()

                aa = np.argwhere(patch > 0)

                if len(aa) > 0:
                    aa[:, 0] = aa[:, 0] + c10
                    aa[:, 1] = aa[:, 1] + c00

                    d_avg = 0.0
                    for i in range(len(aa)):
                        d_avg += frame[aa[i, 0], aa[i, 1]] * depth_scale # depth_frame.get_distance(aa[i, 1], aa[i, 0])  #
                    d_avg /= len(aa)

                    cv2.putText(frame1, text="d={:1.2f}m".format(d_avg), org=(150, 470),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
                                thickness=2, lineType=cv2.LINE_AA)

            cv2.namedWindow('Read image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Read image", frame1)
            cv2.waitKey(1)

        except CvBridgeError as e:
                print(e)


    def msg_to_numpy(self, data):
        try:
            raw_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as err:
            print(err)

        return raw_img

    def draw_rectangle(self, event, x, y, flags, param):
        global pt1, pt2, topLeft_clicked, botRight_clicked

        if event == cv2.EVENT_LBUTTONDOWN:
            rospy.loginfo("EVENT_LBUTTONDOWN")
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


if __name__ == '__main__':
    rospy.init_node('read_rgb')
    ri = ReadImage()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('Exit now')
        cv.destroyAllWindows()
    # sub = rospy.Subscriber("/camera/color/image_raw", Image, self.read_image_callback)
    # rospy.spin()
    # rospy.sleep(1)
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     rate.sleep() # sleep amount of time to have 10Hz