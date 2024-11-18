import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from collections import Counter
characters = []


def selectCharacter(predicted_character):
    if predicted_character != 'NOTHING':
        characters.append(predicted_character)
    else:
        if characters:
            # Count the frequency of each element
            count = Counter(characters)
            # Get the element with the maximum frequency
            most_common_value, _ = count.most_common(1)[0]
            print(most_common_value)
            characters.clear()




model = load_model('/Users/abdullahali/Desktop/Rough-Neural-Network/SignLanguageNeuralNetwork.h5')

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    label_map = {
    0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E', 
    5: 'F',  6: 'G',  7: 'H',  8: 'I',  9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
    25: 'Z', 26: 'SPACE', 27: 'DELETE', 28: 'NOTHING'
}

    ret, frame = cap.read()
    if not ret:
        break


    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            dataBuffer = []
            xCoord = []
            yCoord = []

            # Collect all coordinates for normalization
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                xCoord.append(x)
                yCoord.append(y)

            # Normalize and store coordinates
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                dataBuffer.append(x - min(xCoord))
                dataBuffer.append(y - min(yCoord))

            
            dataBuffer = np.array(dataBuffer).reshape(1, -1)
            predictions = model.predict(dataBuffer, verbose=0)
            predicted_class = np.argmax(predictions[0])
            predicted_character = label_map.get(predicted_class, '?')  # '?' if class not in dictionary

            confidence = predictions[0][predicted_class]
            
            
            cv2.putText(frame, 
                       f"Class: {predicted_character} Conf: {confidence:.2f}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 0, 255), 
                       2)
            
            
            for i, landmark in enumerate(landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            selectCharacter(predicted_character)
    else:
        predicted_character = 'NOTHING'
        cv2.putText(frame, 
                   f"Class: {predicted_character} Conf: 1.00", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 0, 255),  # Red color for "NOTHING"
                   2)
        selectCharacter(predicted_character)

    
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()