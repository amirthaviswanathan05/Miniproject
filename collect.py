import cv2
import mediapipe as mp
import csv
import os
import time

# 1. Setup the "Hand Tracker"
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Create data folder
if not os.path.exists('data'):
    os.makedirs('data')

DATA_PATH = 'data/hand_landmarks.csv'

# 3. Input Label
label = input("Type your sign name (HELLO or LOVE) and press ENTER: ").upper()

cap = cv2.VideoCapture(0)

print(f"Starting camera for '{label}'...")
print("Look at the screen. When you see your hand dots, press 'R' to record.")

recording = False
count = 0

while count < 100:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If the AI sees a hand, draw the skeleton so you know it's working
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Save data ONLY if we pressed 'R'
        if recording:
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([lm.x, lm.y, lm.z])
            
            with open(DATA_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label] + landmarks_list)
            
            count += 1
            color = (0, 0, 255) # Red for recording
        else:
            color = (0, 255, 0) # Green for ready
            
        # Optional: Draw landmarks on screen to verify detection
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        color = (255, 255, 255)

    # Screen Instructions
    status = f"Recording: {count}/100" if recording else "Press 'R' to Start"
    cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Signly Collector", frame)
    
    key = cv2.waitKey(1)
    if key == ord('r'):
        recording = True
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Success! Saved 100 samples for {label}.")