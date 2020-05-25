import pyrealsense2 as rs
import numpy as np

try:
    # This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()

    # Specify dimensions
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)

    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        print("w=%d, h=%d" % (depth.get_width(), depth.get_height()))
        x = np.random.randint(depth.get_width())
        y = np.random.randint(depth.get_height())
        print("d(%d,%d)=%.3f meter" % (x, y, depth.get_distance(x, y)))

    exit(0)

except Exception as e:
    print(e)
    pass
