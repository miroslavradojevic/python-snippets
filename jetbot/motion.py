from jetbot import Robot
import time

robot = Robot()

robot.left(speed=0.3)
time.sleep(0.2)
robot.stop()

robot.forward(0.3)
time.sleep(1.0)
robot.stop()

robot.backward(0.3)
time.sleep(1.0)
robot.stop()