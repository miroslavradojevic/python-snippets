import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        # images = np.hstack((color_image, depth_image)) # depth_colormap

        # Show images
        cv2.namedWindow('RGB', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB', color_image)

        cv2.namedWindow('D', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('D', cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))

        cv2.namedWindow('D2', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('D2', depth_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Stop streaming
    pipeline.stop()