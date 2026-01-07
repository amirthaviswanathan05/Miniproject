import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# 1. SETUP HOLISTIC
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

if not os.path.exists('data'): os.makedirs('data')
DATA_PATH = 'data/holistic_data.csv'

label = input("Type the SIGN NAME: ").upper()
cap = cv2.VideoCapture(1400)
recording = False
count = 0
SAMPLES = 300 # More samples needed for Holistic

while count < SAMPLES:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    # Drawing for feedback
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if recording:
        # DATA EXTRACTION
        def get_landmarks(res):
            return [val for lm in res.landmark for val in [lm.x, lm.y, lm.z]] if res else [0.0] * (33 * 3 if hasattr(res, 'landmark') and len(res.landmark) == 33 else 21 * 3)

        # We focus on Pose (shoulders/arms) and both hands
        pose = [val for lm in (results.pose_landmarks.landmark if results.pose_landmarks else []) for val in [lm.x, lm.y, lm.z]]
        if not pose: pose = [0.0] * (33 * 3)
        
        lh = [val for lm in (results.left_hand_landmarks.landmark if results.left_hand_landmarks else []) for val in [lm.x, lm.y, lm.z]]
        if not lh: lh = [0.0] * (21 * 3)
        
        rh = [val for lm in (results.right_hand_landmarks.landmark if results.right_hand_landmarks else []) for val in [lm.x, lm.y, lm.z]]
        if not rh: rh = [0.0] * (21 * 3)

        row = [label] + pose + lh + rh
        with open(DATA_PATH, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        
        count += 1
        cv2.putText(frame, f"RECORDING: {count}/{SAMPLES}", (10, 50), 1, 2, (0,0,255), 2)

    cv2.imshow("Signly Holistic Collector", frame)
    key = cv2.waitKey(1)
    if key == ord('r'): recording = True
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()