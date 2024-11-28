import requests
import nltk
from nltk.tokenize import word_tokenize
import language_tool_python
import time
#make sure your system has java when running this for language_tool_python
# Ensure NLTK resources are downloaded
nltk.download('punkt')

def text_to_speech(text):
    """Sends the text to the raspberry pi to speak"""
    try:
        url = "http:ip:8888/say"
        data = {"text": text}
        response = requests.post(url, json=data)
        print(f"Sent! Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occured while sending the request {e}")
  

def refine_sentence_with_grammar(sentence):
    """Uses LanguageTool for grammar correction."""
    tool = language_tool_python.LanguageTool('en-US')
    corrected_sentence = tool.correct(sentence)
    return corrected_sentence

def formulate_sentence(wordList, use_grammar_check=True):
    """Formulates and refines a sentence from a list of words."""
    sentence = " ".join(wordList)
    if not sentence.endswith(('.', '!', '?')):
        sentence += '.'  # Add punctuation if missing at the end
    # Apply grammar correction
    if use_grammar_check:
        sentence = refine_sentence_with_grammar(sentence)
    return sentence.capitalize()

def intake(character, workingString, wordList, outputSentence=False):
    """Processes characters, formulates words, and optionally converts to speech."""
    if character == "SPACE":
        if workingString:  # Add non-empty word to wordList
            wordList.append(workingString)
        workingString = ""
    elif character == "DELETE":
        workingString = workingString[:-1]
    elif character != "NOTHING":
        workingString += character

    if outputSentence and wordList:
        # Formulate and refine the sentence
        sentence = formulate_sentence(wordList, use_grammar_check=True)
        text_to_speech(sentence)  # Speak the sentence
        wordList.clear()  # Clear word list after speaking
    return workingString, wordList

#Test Cases

def run_tests():
    workingString = ""
    wordList = []

    # Test Case 1: Basic Sentence Formation
    print("\nTest Case 1: Basic Sentence Formation")
    workingString, wordList = intake("h", workingString, wordList)
    workingString, wordList = intake("e", workingString, wordList)
    workingString, wordList = intake("l", workingString, wordList)
    workingString, wordList = intake("l", workingString, wordList)
    workingString, wordList = intake("o", workingString, wordList)
    workingString, wordList = intake("SPACE", workingString, wordList)
    workingString, wordList = intake("w", workingString, wordList)
    workingString, wordList = intake("o", workingString, wordList)
    workingString, wordList = intake("r", workingString, wordList)
    workingString, wordList = intake("l", workingString, wordList)
    workingString, wordList = intake("d", workingString, wordList)
    workingString, wordList = intake("SPACE", workingString, wordList)
    print("Intermediate Word List:", wordList)
    workingString, wordList = intake("NOTHING", workingString, wordList, outputSentence=True)

    # Test Case 2: Handling Delete
    print("\nTest Case 2: Handling Delete")
    workingString, wordList = intake("DELETE", workingString, wordList)
    workingString, wordList = intake("a", workingString, wordList)
    workingString, wordList = intake("SPACE", workingString, wordList)
    print("Intermediate Word List:", wordList)
    workingString, wordList = intake("NOTHING", workingString, wordList, outputSentence=True)

    # Test Case 3: Grammar Correction
    print("\nTest Case 3: Grammar Correction")
    wordList = ["this", "is", "a", "test", "sentense"]
    sentence = formulate_sentence(wordList, use_grammar_check=True)
    print("Grammar Corrected Sentence:", sentence)

    # Test Case 4: Multiple Spaces
    print("\nTest Case 4: Multiple Spaces")
    workingString, wordList = intake("SPACE", workingString, wordList)
    workingString, wordList = intake("SPACE", workingString, wordList)
    workingString, wordList = intake("NOTHING", workingString, wordList, outputSentence=True)

    # Test Case 5: Empty Input
    print("\nTest Case 5: Empty Input")
    workingString, wordList = intake("NOTHING", workingString, wordList)
    print("Word List After Empty Input:", wordList)

    # Test Case 6: Sentence with Numbers
    print("\nTest Case 6: Sentence with Numbers")
    wordList = ["test", "123"]
    sentence = formulate_sentence(wordList, use_grammar_check=True)
    print("Sentence with Numbers:", sentence)
    text_to_speech(sentence)

    # Test Case 7: Complex Sentence
    print("\nTest Case 7: Complex Sentence")
    wordList = ["how", "are", "you", "today", "?"]
    sentence = formulate_sentence(wordList, use_grammar_check=True)
    print("Complex Sentence:", sentence)
    text_to_speech(sentence)

    # Test Case 8: Sentence with Punctuation
    print("\nTest Case 8: Sentence with Punctuation")
    wordList = ["this", "is", "a", "test", "."]
    sentence = formulate_sentence(wordList, use_grammar_check=True)
    print("Sentence with Punctuation:", sentence)
    text_to_speech(sentence)

    # Test Case 9: Long Sentence with Grammar Correction
    print("\nTest Case 9: Long Sentence with Grammar Correction")
    wordList = ["i", "am", "going", "to", "the", "markit"]
    sentence = formulate_sentence(wordList, use_grammar_check=True)
    print("Long Corrected Sentence:", sentence)
    text_to_speech(sentence)

#run_tests()
