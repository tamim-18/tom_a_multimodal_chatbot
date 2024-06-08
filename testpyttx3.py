import pyttsx3

# Initialize the TTS engine
tts_engine = pyttsx3.init()

# Set properties (optional)
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Text to be spoken
text = "Hello, this is a test of the text-to-speech functionality."

# Use TTS to speak the text
tts_engine.say(text)
tts_engine.runAndWait()
