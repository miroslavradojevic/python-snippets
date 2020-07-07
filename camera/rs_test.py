import pyrealsense2 as rs

pipe = rs.pipeline()
profile = pipe.start()

try:
  for i in range(0, 5):
    frames = pipe.wait_for_frames()
    print("\n", len(frames), " frames ", type(frames))

    color_frame = frames.get_color_frame()

    print(type(color_frame), "h=", color_frame.get_height(), "w=", color_frame.get_width())

    for f in frames:
        print(f.profile)


finally:
    pipe.stop()