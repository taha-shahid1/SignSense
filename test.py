import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model('SignLanguageNeuralNetwork.h5')

# Hand detection setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Dictionary that maps predicted labels to corresponding sign
label_to_letter = {i: chr(ord('A') + i) for i in range(26)}  # 0->'a', 1->'b', ..., 25->'z'
# Still need to manually add the remaining non-letter signs

# Function that takes in hand landmarks from MediaPipe containing 3D coordinates of points on a captured hand
# Returns: dataBuffer (NumPy array), ready for the model to predict
def process_landmarks(landmarks):
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
    return dataBuffer

# Function that takes in processed landmark data and feeds it to the model
# Returns: a letter (string) which the model predicts along with the confidence (float) the model has
def predict_gesture(dataBuffer):
    predictions = model.predict(dataBuffer, verbose=0)  # Get probabilities for each gesture
    predictedLabel = np.argmax(predictions[0])  # Get the index of the highest probability
    letter = label_to_letter.get(predictedLabel, '?')
    confidence = predictions[0][predictedLabel]
    return letter, confidence

# Start the video capture locally, change for raspberry pi
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get the model's predictions
            dataBuffer = process_landmarks(landmarks)
            letter, confidence = predict_gesture(dataBuffer)                     
            
            # Display the model's predictions
            cv2.putText(frame, 
                       f"Class: {letter} Conf: {confidence:.2f}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)
            
            # Draw the landmarks
            for i, landmark in enumerate(landmarks.landmark):
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()