import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Access system volume controls
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume.GetVolumeRange()  # (-65.25, 0.0)
min_vol, max_vol = vol_range[0], vol_range[1]

# Start Video Capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert image to RGB
        frame = cv2.flip(frame, 1)  # Mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand tracking
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions for thumb and index finger
                landmarks = hand_landmarks.landmark
                h, w, _ = frame.shape
                thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
                index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)

                # Draw circles on thumb and index finger
                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)

                # Calculate distance between thumb and index finger
                distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

                # Convert distance to volume range
                vol = np.interp(distance, [30, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)

                # Display Volume Level
                vol_bar = np.interp(distance, [30, 200], [400, 150])
                cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)

        cv2.imshow("Hand Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
