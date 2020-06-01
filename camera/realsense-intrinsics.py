import pyrealsense2 as rs
# import numpy as np
# import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start pipeline with defined configuration
cfg = pipeline.start(config)

try:
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    
    print(type(intr))
    print(intr)

finally:
    # Stop streaming
    pipeline.stop()