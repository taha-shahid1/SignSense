import mediapipe as mp
import pickle
import os
import cv2 as cv

# Takes images from dataset folder, uses mediapipe to extract the coordinates of landmarks on the hands in the image, and stores the coordinates in a binary file to be used in training the model.

DATA_DIR = '/path/to/the/dataset'

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
data = []
labels = []

for dir in os.listdir(DATA_DIR):
    # Skip non-directory files like .DS_Store
    dirPath = os.path.join(DATA_DIR, dir)
    if not os.path.isdir(dirPath):
        continue

    for imgPath in os.listdir(dirPath):
        dataBuffer = []
        xCoord = []
        yCoord = []

        img = cv.imread(os.path.join(dirPath, imgPath))
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # First collect all coordinates for normalization
                for i in range(len(landmarks.landmark)):
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y
                    xCoord.append(x)
                    yCoord.append(y)
                
                # Then normalize and store them, as normalizing data scales input features to a smaller consistent range, improving training speed and preventing large values from dominating (preventing very high or very low weights being assigned to a neuron)
                for i in range(len(landmarks.landmark)):
                    x = landmarks.landmark[i].x
                    y = landmarks.landmark[i].y
                    dataBuffer.append(x - min(xCoord))
                    dataBuffer.append(y - min(yCoord))
            
            data.append(dataBuffer)
            labels.append(dir)


# Stores landmarks into a binary file (picklefile) to be used in training the neural network
picklefile = open('dataset.pickle', 'wb')
pickle.dump({'labels': labels, 'data': data}, picklefile)
picklefile.close()