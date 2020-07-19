
import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = "/home/miro/stack/nuctech/jetbot_obstacle_avoidance"


if __name__ == "__main__":
    print(os.listdir(data_dir))
    print(len(sys.argv), " ", sys.argv[0])
    print(help(ImageDataGenerator))