import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to listen to the microphone and recognize speech
def recognize_speech():
    # Use the microphone as the source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for your speech...")
        
        # Capture the audio
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google's speech recognition
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.RequestError:
        print("Sorry, there was an error with the speech service.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    recognize_speech()



#pip install SpeechRecognition pyaudio
