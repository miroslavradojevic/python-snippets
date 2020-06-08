# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py

import pyrealsense2 as rs
# import numpy as np
# import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

cfg = pipeline.start(config)

try:
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

    print(profile.as_video_stream_profile())
    print("cx = ", intr.ppx, " cy = ", intr.ppy)

finally:
    pipeline.stop()