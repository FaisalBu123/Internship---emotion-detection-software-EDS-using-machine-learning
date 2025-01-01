# let us carry out speech recognition as well as pitch and sentiment analysis
# This is just a simple piece of code, complex code was not in the scope of this project

import librosa
import cv2
import speech_recognition as sr
from moviepy import VideoFileClip
from textblob import TextBlob
import numpy as np

# This function will allow me to extract speech from video file
def extract_speech(video_path):
    recognizer = sr.Recognizer()
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_file = "audio.wav"
    audio_clip.write_audiofile(audio_file)

    # Recognize speech from the extracted audio
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)

    # Perform speech recognition
    try:
        speech_text = recognizer.recognize_google(audio)   # use google speech recognition
        return speech_text.strip()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

# measure sentiment using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0.2:
        return "Positive"
    elif sentiment_score < -0.2:
        return "Negative"
    else:
        return "Neutral"


# analysing pitch using Librosa
def analyze_pitch(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract pitch using librosa's piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Calculate the mean pitch, ignoringg any zeros
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    mean_pitch = np.mean(pitch_values)

    # Categorize pitch levels
    if mean_pitch > 95:
        return "High Pitch"
    elif mean_pitch < 90:
        return "Low Pitch"
    else:
        return "Medium Pitch"


# call the functions
video_path = "Internship/V1/P/S.mp4"
extracted_speech = extract_speech(video_path)
print("Extracted Speech:", extracted_speech)

sentiment = analyze_sentiment(extracted_speech)
print("Sentiment:", sentiment)

audio_path = "audio.wav"
pitch_analysis = analyze_pitch(audio_path)
print("Pitch Analysis:", pitch_analysis)

if (sentiment == "Positive" and (pitch_analysis == "High Pitch" or pitch_analysis == "Neutral Pitch")) \
    or (sentiment == "Neutral" and pitch_analysis == "High Pitch"):
    print("Overall speech analysis is Positive")
else:
    print("Overall speech analysis is Negative")
