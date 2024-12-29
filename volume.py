import cv2
import pyautogui
import mediapipe as mp
import pickle

# Load the model - Assuming you have a trained model stored in 'model.p'
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_hands_constants = mp.solutions.hands.HandLandmark

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {
    11: 'palm', 12: 'index in right', 13: 'index in left', 14: 'Three fingers', 15: 'Three Finger in right'
}

# Action mapping 
action_mapping = {
    'Three fingers': 'volume_up',
    'Three Finger in right': 'volume_down',
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_y = hand_landmarks.landmark[mp_hands_constants.THUMB_TIP].y
            index_finger_y = hand_landmarks.landmark[mp_hands_constants.INDEX_FINGER_TIP].y
            middle_finger_y = hand_landmarks.landmark[mp_hands_constants.MIDDLE_FINGER_TIP].y

            if thumb_y < index_finger_y and thumb_y < middle_finger_y:
                hand_gesture = 'volume_up'
            elif thumb_y > index_finger_y and thumb_y > middle_finger_y:
                hand_gesture = 'volume_down'
            else:
                hand_gesture = 'other'

            if hand_gesture in action_mapping:
                pyautogui.press(action_mapping[hand_gesture])

            # Display action mapping on the screen
            cv2.putText(frame, action_mapping.get(hand_gesture, 'Unknown'),
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
