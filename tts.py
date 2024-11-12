import pyttsx3 # Pip install first

def text_to_speech(text):
    # Initialize the Text-to-Speech engine
    engine = pyttsx3.init()
    
    # Set properties before adding text
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()

# Get input from the user
user_input = "hello"
text_to_speech(user_input)
