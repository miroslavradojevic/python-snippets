import pygame
import sys
import time

# https://www.pygame.org/docs/ref/pygame.html
# https://shop.sb-components.co.uk/blogs/piarm-lessons/lesson-3-controlling-using-joystick

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystics were found")
    sys.exit()


joy = pygame.joystick.Joystick(0)
joy.init()

# joy.get_init()

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
        # event types: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:
                command["axis_0"] = round(event.value, 6)
            elif event.axis == 1:
                command["axis_1"] = round(event.value, 6)
        elif event.type == pygame.JOYBUTTONDOWN and event.button == 3:
            command["exit"] = True

    return command


while True:
    joystick_events = pygame.event.get()  # pygame.event.pump()

    joystic_command = process_joystic_events(joystick_events)

    sys.stdout.write("\r{0} {1: <60}".format(time.ctime(), str(joystic_command)))
    sys.stdout.flush()

    if "exit" in joystic_command:
        if joystic_command["exit"]:
            break # get out of the loop with button 3

    time.sleep(0.1)

print("\nExit...")
pygame.joystick.quit()
