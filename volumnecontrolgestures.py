import cv2
import time
import HandTrackingModule as htm
from mediapipe import *
# Open webcam
vid = cv2.VideoCapture(0)
vid.set(3,250)
vid.set(4,250)

detector=htm.handDetector()
# Check if the webcam opened successfully
if not vid.isOpened():
    print("Error: Could not open webcam.")
    exit()
ftime=0
while True:
    success, image = vid.read()
    image=detector.findHands(image)
    ctime=time.time()
    fps=1/(ctime-ftime)
    ftime=ctime
    # Check if the frame was captured successfully
    if not success:
        print("Error: Could not read frame.")
        break

    cv2.putText(img=image,text=f'FPS: {int(fps)}',
        org=(30, 50),  # Position for the text (x, y)
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font
        fontScale=1,  # Font size
        color=(255, 0, 0),  # Color in BGR (blue here)
        thickness=2,  # Line thickness
        lineType=cv2.LINE_AA )
    cv2.imshow("Webcam Feed", image)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()
