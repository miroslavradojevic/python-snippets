# Use opencv to connect to USB camera
import cv2
import time
from datetime import datetime

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 32


# MACOS AND LINUX: *'XVID' (MacOS users may want to try VIDX as well)
# WINDOWS *'VIDX'
writer = cv2.VideoWriter("capture_" + time.strftime("%Y%m%d-%H%M%S") + ".mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Write the video
    writer.write(frame)
    print("{}".format(datetime.now()))

    cv2.putText(frame, text='Recording...',
                org=(int(0.7*width), int(0.1*height)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255), thickness=1,
                lineType=cv2.LINE_AA)

    cv2.putText(frame, text='Press "q" to exit',
                org=(0, int(0.95 * height)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255), thickness=1,
                lineType=cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Quit with the "q" button on a keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
