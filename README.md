# SignSense

**SignSense** is a real-time American Sign Language (ASL) letter recognition system developed as part of the SE101 course at the University of Waterloo. The system uses a Raspberry Pi Zero W and a camera module to capture hand orientation landmarks, enabling the prediction of ASL letters in real-time.

## Installation

### Requirements

Before running the project, make sure you have the following:

- **Raspberry Pi Zero W** with a camera module and audio output
- Python 3.10.x
- TensorFlow
- Flask
- OpenCV
- NumPy
- MediaPipe
- LanguageTool

### Setup

1. **On the Raspberry Pi**:
   - Clone the repository and install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - Start the Flask server on the Raspberry Pi by running:
     ```bash
     python flaskie.py
     ```

2. **On an external computer** (same network as the Raspberry Pi):
   - Clone the repository and install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the `test.py` script to start receiving and displaying the ASL predictions:
     ```bash
     python test.py
     ```

## Usage

1. Once `flaskie.py` is running on the Raspberry Pi, and `test.py` is running on the external computer, the camera module on the Raspberry Pi will begin capturing signs.
2. The system will stream the captured video to the external computer, predict the ASL letter, and send the result back to the Raspberry Pi for text-to-speech output.
