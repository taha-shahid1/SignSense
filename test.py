import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from updated_main import intake, formulate_sentence, text_to_speech, refine_sentence_with_grammar

# Load the model
model = load_model('SignLanguageNeuralNetwork.h5')

# Hand detection setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Store the previously used characters in a frequency list
last_characters = [0] * 28

# Dictionary that maps predicted labels to corresponding sign
label_to_letter = {i: chr(ord('A') + i) for i in range(26)}  # 0->'a', 1->'b', ..., 25->'z'
label_to_letter[26] = "SPACE"
label_to_letter[27] = "DELETE"
label_to_letter[28] = "NOTHING"

# Function that takes in the last frame's detection and returns a character if the nothing has been met
def selectCharacter(predicted_class):
    global last_characters
    threshold = 10 # Don't return a character if the minimum has not been reached
    
    # If the detected sign was a regular letter, update the frequency table
    if predicted_class != 28: # NOTHING
        last_characters[predicted_class] += 1
        return None
    
    # If the nothing character is sent and the threshold is met, print and return the most likely letter
    max_count = max(last_characters)
    if max_count >= threshold:
        max_class = last_characters.index(max_count)
        last_characters = [0] * 28
        return label_to_letter.get(max_class, "?") # '?' if not found, shouldn't happen    

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

streamURL = "http://172.20.10.2:8888/video_feed"
cap = cv2.VideoCapture(streamURL)

workingString = ""
wordList = []
recent_predictions = []
count = 0

while cap.isOpened():
    count += 1
    if count%2:
        continue
    ret, frame = cap.read()
    if not ret:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    letter = None

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
            
            
            
            letter = selectCharacter(next((k for k, v in label_to_letter.items() if v == letter), None))
    else: # If no landmarks were created
        letter, confidence = "NOTHING", 1
        
        # Display the model's predictions
        cv2.putText(frame, 
                    f"Class: {letter} Conf: {confidence:.2f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2)
        letter = selectCharacter(28)
    
    if letter:
            recent_predictions.append(letter)
            if len(recent_predictions) > 2:
                recent_predictions.pop(0)
            print("Recent Predictions:", recent_predictions)  # Debugging
            if recent_predictions == ["SPACE", "SPACE"]:
                if wordList:
                    sentence = formulate_sentence(wordList, use_grammar_check=True)
                    print("Output Sentence:", sentence)
                    text_to_speech(sentence)
                    wordList.clear()
            if letter != "NOTHING":
                workingString, wordList = intake(letter, workingString, wordList, outputSentence=False)

        
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()