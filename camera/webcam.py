# Use opencv to capture information from (usb, web) camera
# use python opencv to access camera image and show/save it
import cv2

# Connect to computer default camera
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("frame dimensions, w=%d, h=%d" % (width, height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.putText(gray, text='Press "q" to exit', org=(0, int(0.95*height)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 255, 255), thickness=1,
                lineType=cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', gray)

    # Quit with the "q" button on a keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release capture and close all windows
cap.release()
cv2.destroyAllWindows()
