import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyautogui
import time

# Load the model
model_path = 'C:/Users/RIYA/Desktop/rk/amisha ml/model.p'
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']               

# Initialize webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    11: 'palm', 12: 'index in right', 13: 'index in left', 14: 'Three fingers', 15: 'Three Finger in right'
}

# Action mapping 
action_mapping = {
    'palm': 'Play',
    'index in right': 'Next song',
    'index in left': 'Previous song',
    'Three fingers': 'Increase Volume',
    'Three Finger in right': 'Reduce volume',
}

def play_next_song():
    pyautogui.hotkey('ctrl', 'f')

def play_previous_song():
    pyautogui.hotkey('ctrl', 'b')

def decrease_volume():
    pyautogui.press('f2')  # Press F2 to decrease volume
    time.sleep(1)

def increase_volume():
    pyautogui.press('f3')  # Press F3 to increase volume
    time.sleep(1)

def detect_gesture(frame, landmarks):
    thumb_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    palm = landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

    thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
    index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
    palm_x, palm_y = int(palm.x * frame.shape[1]), int(palm.y * frame.shape[0])

    fingers = [1 if lm in [thumb_tip, index_tip] else 0 for lm in [thumb_tip, index_tip, landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP],
            landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP], landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]]]

    # Check for specific gesture related to playing next or previous song
    if abs(thumb_x - index_x) < 30:
        # Check if thumb and index finger are close to each other horizontally
        if thumb_y < index_y:
            # Check if thumb is above index finger and no other fingers are raised
            if thumb_y < palm_y and sum(fingers) == 2:  
                print("Next Song Gesture Detected")
                play_next_song()
                return  # Exit the function to prioritize playing next song
        elif thumb_y > index_y:
            # Check if thumb is below index finger and no other fingers are raised
            if thumb_y > palm_y and sum(fingers) == 2:  
                print("Previous Song Gesture Detected")
                play_previous_song()
                return  # Exit the function to prioritize playing previous song

    # Check if the palm is shown
    if palm_y > index_y and palm_y > thumb_y:
        # Toggle play/pause
        pyautogui.press('space')
        print("Play/Pause Gesture Detected")

    # Check for other gestures
    if fingers == [0, 1, 1, 1, 0]:  # 3 fingers shown, middle one then increase the volume
        increase_volume()
        print("Increase Volume Gesture Detected")
    elif fingers == [1, 1, 1, 0, 0] and palm_x > index_x and palm_x > thumb_x:  # 3 fingers shown, directed towards right, decrease the volume
        decrease_volume()
        print("Decrease Volume Gesture Detected")

def gesture_command():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                detect_gesture(frame, landmarks)

        cv2.imshow('Gesture Detection for Media Player Control', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_command()
