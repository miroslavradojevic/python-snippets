import pygame
import sys

from jetbot import Robot

# install pygame dependencies
# https://stackoverflow.com/questions/7652385/where-can-i-find-and-install-the-dependencies-for-pygame
# sudo apt-get install python-dev \
# libsdl-image1.2-dev \
# libsdl-mixer1.2-dev \
# libsdl-ttf2.0-dev \
# libsdl1.2-dev \
# libsmpeg-dev \
# python-numpy \
# subversion \
# libportmidi-dev \
# ffmpeg \
# libswscale-dev \
# libavformat-dev \
# libavcodec-dev

# libfreetype6-dev
# sudo pip3 install pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystics were found")
    sys.exit()

joy = pygame.joystick.Joystick(0)
joy.init()

joystick_info = dict()
joystick_info["ID"] = joy.get_id()
joystick_info["Name"] = joy.get_name()
joystick_info["NrAxes"] = joy.get_numaxes()
joystick_info["NrBalls"] = joy.get_numballs()
joystick_info["NrButtons"] = joy.get_numbuttons()
joystick_info["NrHats"] = joy.get_numhats()
joystick_info["NrAxes"] = joy.get_numaxes()

print(joystick_info)


def process_joystic_events(events):
    command = dict()
    for event in events:
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:
                command["axis_0"] = round(event.value, 6)
            elif event.axis == 1:
                command["axis_1"] = round(event.value, 6)
        elif event.type == pygame.JOYBUTTONDOWN and event.button == 3:
            command["exit"] = True

    return command


def motor_coeff_left_compute(command_value):
    if command_value <= 0:
        return 1.0
    elif command_value <= 1.0:
        return 1.0 - command_value
    else:
        return 0.0


def motor_coeff_right_compute(command_value):
    if command_value <= -1.0:
        return 0.0
    elif command_value <= 0.0:
        return command_value + 1.0
    else:
        return 1.0


def motor_coeff_forward_compute(command_value):
    if command_value <= -1.0:
        return 1.0
    elif command_value <= 1.0:
        return -1.0 * command_value
    else:
        return 1.0


def show_usage():
    print("Use joystick to move robot")
    print("Axis 0: left - right")
    print("Axis 1: forward - backward")
    print("Button 3: exit")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        show_usage()
        sys.exit()

    motor_coeff_left = 0
    motor_coeff_right = 0
    motor_coeff_forward = 0
    try:
        max_velocity = float(sys.argv[1])
    except ValueError:
        print("Argument ", sys.argv[1], " is not a float") 
        sys.exit()

    robot = Robot()

    while True:
        joystick_events = pygame.event.get()
        joystic_command = process_joystic_events(joystick_events)

        # joystic_command has "exit", "axis_0" and "axis_1" fields
        if "exit" in joystic_command:
            if joystic_command["exit"]:
                robot.stop()
                break  # get out of the loop with button 3

        if "axis_0" in joystic_command:
            motor_coeff_left = motor_coeff_left_compute(joystic_command["axis_0"])
            motor_coeff_right = motor_coeff_right_compute(joystic_command["axis_0"])
            
        if "axis_1" in joystic_command:
            motor_coeff_forward = motor_coeff_forward_compute(joystic_command["axis_1"])

        motor_left = motor_coeff_left * motor_coeff_forward * max_velocity
        motor_right = motor_coeff_right * motor_coeff_forward * max_velocity

        # set speeds to robot wheels
        robot.set_motors(motor_left, motor_right)
        # robot.left_motor.value = motor_left
        # robot.right_motor.value = motor_right

        sys.stdout.write("\rL={0:+1.3f} R={1:+1.3f}    ".format(motor_left, motor_right))
        sys.stdout.flush()


print("\nExiting...")
pygame.joystick.quit()
