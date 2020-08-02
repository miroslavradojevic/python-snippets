#!/usr/bin/env python
# training data for collision detection
# use joystick and RPi_v2 camera
# button left/right would save current camera capture as negative/positive (there is obstacle/no obstacle found)
import pygame
import sys
import os
import time
import cv2
from datetime import datetime
from matplotlib.image import imread
import numpy as np
from cam_rpiv2 import CameraRPiv2

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    exit("No joystics were found")

joy = pygame.joystick.Joystick(0)
joy.init()

def process_joystic_events(events):
    command = dict()
    for event in events:
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == 6:
                command["L"] = "ON"
            elif event.button == 7:
                command["R"] = "ON"
            elif event.button == 3:
                command["exit"] = True

    return command

def show_usage():
    print("Use joystick to capture camera")
    print("\nUsage: python3 " + __file__)
    print("Button L: obstacle")
    print("Button R: no obstacle")


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

if __name__ == "__main__":
    
    if len(sys.argv) != 1:
        show_usage()
        sys.exit()


    d_name = os.path.abspath(os.path.dirname(__file__))
    f_name = os.path.splitext(os.path.basename(__file__))[0]
    
    d_out = os.path.join(d_name, f_name)
    create_dir(d_out)

    d_out_0 = os.path.join(d_name, f_name, "0")
    create_dir(d_out_0)

    d_out_1 = os.path.join(d_name, f_name, "1")
    create_dir(d_out_1)

    cam = CameraRPiv2()

    try:
        print("Press L/R joystick button...\n")
        while True:
            joystick_events = pygame.event.get()
            joystic_command = process_joystic_events(joystick_events)

            if bool(joystic_command):
                sys.stdout.write("\rJoystick command: {}".format(joystic_command))
                sys.stdout.flush()

                if "exit" in joystic_command:
                    print("\nExiting...")
                    break  # get out of the loop with button 3

                if "L" in joystic_command:
                    img_path = os.path.join(d_out_0, datetime.now().strftime("%Y%m%d-%H%M%S-%f") +'.jpg')
                    # cam.value <class 'numpy.ndarray'> (2464, 3280, 3) uint8 0 255
                    cv2.imwrite(img_path, cam.value)
                    print("\n", img_path)

                if "R" in joystic_command:
                    img_path = os.path.join(d_out_1, datetime.now().strftime("%Y%m%d-%H%M%S-%f") +'.jpg')
                    cv2.imwrite(img_path, cam.value)
                    print("\n", img_path)

            time.sleep(0.001)

    except KeyboardInterrupt:
        cam.stop()
        pygame.joystick.quit()
        pygame.quit()
            
    cam.stop()
    pygame.joystick.quit()
    pygame.quit()