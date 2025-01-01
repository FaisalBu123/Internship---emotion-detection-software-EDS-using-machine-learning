# let us carry out speech recognition as well as pitch and sentiment analysis
# This is just a simple piece of code, complex code was not in the scope of this project

import librosa
import speech_recognition as sr
from moviepy import VideoFileClip
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# This function will allow me to extract speech from video file
def extract_speech(video_path):
    recognizer = sr.Recognizer()
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_file = "audio.wav"
    audio_clip.write_audiofile(audio_file)

    # Recognise speech from the extracted audio
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

# Generate a labeled dataset using TextBlob's sentiment analysis
def generate_labeled_data(texts):
    labeled_data = []
    for text in texts:
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        
        if sentiment_score > 0.2:
            sentiment = "Positive"
        elif sentiment_score < -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        labeled_data.append((text, sentiment))
    return labeled_data

# Create a model pipeline with vectorizer and logistic regression
def train_sentiment_model(train_data):
    X_train, y_train = zip(*train_data)
    model = make_pipeline(CountVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)
    return model

# Analyze sentiment using the trained model
def analyze_sentiment_ml(text, model):
    sentiment = model.predict([text])
    return sentiment[0]

# analysing pitch using Librosa
def analyze_pitch(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract pitch using librosa's piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Calculate the mean pitch, ignoring any zeros
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    mean_pitch = np.mean(pitch_values)

    # Categorize pitch levels
    if mean_pitch > 95:
        return "High Pitch"
    elif mean_pitch < 90:
        return "Low Pitch"
    else:
        return "Medium Pitch"


# Large block of text/phrases so I can train the model (this can be expanded)
texts = [
    "I love this movie!", "This is terrible.", "I'm feeling happy today.", 
    "I feel great!", "This is a bad experience.", "I feel neutral about this.",
    "What a wonderful day!", "I am excited for tomorrow.",
    "The weather is fine, not too hot or cold.", "I'm not happy with my service.",
    "I'm so thrilled with this purchase!", "I absolutely hate this!", "I'm very satisfied with the results.",
    "This is the best thing I've ever done.", "I'm feeling very excited!", "I regret buying this.",
    "I can't wait to do this again.", "What a disappointing experience!", "This is a fantastic opportunity!",
    "I'm so angry right now.", "I feel really good about this.", "I would never recommend this to anyone.",
    "This is exactly what I was hoping for!", "I can't believe how bad this is.", "I'm feeling a bit sad.",
    "This is amazing, I'm so impressed!", "I feel indifferent about this.", "Such an amazing experience!",
    "This was a waste of time.", "I'm in love with this!", "I don't like it at all.", "I feel so proud.",
    "What a lovely surprise!", "This is just awful.", "I can't stop smiling, this is awesome!",
    "It’s alright, not great, not terrible.", "I’m completely disillusioned with this.", "I feel really proud of what I've done.",
    "I’m so happy with my decision!", "This is absolutely horrible.", "I think it’s decent, but not great.",
    "I’m really looking forward to the weekend.", "This is a total disaster.", "I feel relieved, it’s over.",
    "I’m so grateful for this experience.", "I feel disappointed with the outcome.", "This was a positive change.",
    "I'm content with this.", "I don't feel anything about it.", "I feel optimistic about the future.",
    "I feel hopeless about this situation.", "This product is wonderful!", "It’s the worst thing I’ve tried.",
    "I feel really energized after this!", "I’m not sure how I feel about it yet.", "I feel comfortable with my decision.",
    "I feel extremely stressed.", "I'm so proud of my work!", "This isn’t what I expected at all.",
    "It’s an absolute joy to use this.", "I really don’t like the service here.", "This could have been so much better.",
    "I'm feeling indifferent, not happy, not sad.", "I’m thrilled with the results!", "It’s an okay experience.",
    "This is the worst mistake I’ve ever made.", "I’ve never been more satisfied.", "I’m frustrated with the outcome.",
    "I’m overjoyed!", "I’m really let down by this.", "It was a wonderful surprise!", "This is a great accomplishment.",
    "I can't express how much I dislike this.", "This is absolutely fantastic!", "I feel very anxious about this.",
    "I’m unsure if this is a good idea.", "What a brilliant experience!", "This is such a big letdown.",
    "I am so relieved!", "I’m not excited about it.", "This is just perfect!", "I don’t care much about it.",
    "I feel so at peace with this.", "I’m overwhelmed with joy!", "What a frustrating day.",
    "I’m impressed with the results.", "This is just okay, not fantastic.", "I’m utterly disappointed.",
    "I feel so happy I made this choice.", "This is a big disappointment.", "It was a lovely experience overall.",
    "This is a complete nightmare.", "I’m so happy it worked out!", "I have mixed feelings about it.",
    "What a successful outcome!", "I feel pretty bad about this.", "I don’t feel great about this at all.",
    "I’m very enthusiastic about it.", "It’s not as great as I thought it would be.", "I’m very satisfied with this purchase.",
    "I’m really unhappy with my decision.", "What a breathtaking experience!", "I’m so proud of what I achieved.",
    "It’s okay, but it could be better.", "This has been a pleasant surprise!", "I feel exhausted after this.",
    "I feel inspired by this experience.", "I feel defeated by this situation.", "I’m not enjoying this at all.",
    "I feel amazing!", "I’ve never felt so good.", "I have mixed emotions about this.", "This was a pleasant experience.",
    "It was okay, but nothing extraordinary.", "I feel unsure about this decision.", "I feel rejuvenated after this!",
    "This is the worst thing I've ever seen!", "I’m so thrilled with how things turned out.", "I can’t stop laughing, this is so fun!"
]


# Generate labeled data using TextBlob sentiment. This will attach labels to each of the sentences above.
# the labels are positive, negative, and neutral sentiment. This data is then used to train our model so it can recognise sentiment of new text 
# Note that this technique is not perfect as textblob can sometimes make some errors, but it works in this case
train_data = generate_labeled_data(texts)

# Train the sentiment model
sentiment_model = train_sentiment_model(train_data)

# Call the functions
video_path = "Internship/V1/P/S.mp4"
extracted_speech = extract_speech(video_path)
print("Extracted Speech:", extracted_speech)

sentiment = analyze_sentiment_ml(extracted_speech, sentiment_model)
print("Sentiment:", sentiment)

audio_path = "audio.wav"
pitch_analysis = analyze_pitch(audio_path)
print("Pitch Analysis:", pitch_analysis)

if (sentiment == "Positive" and (pitch_analysis == "High Pitch" or pitch_analysis == "Neutral Pitch")) \
    or (sentiment == "Neutral" and pitch_analysis == "High Pitch"):
    print("Overall speech analysis is Positive")
else:
    print("Overall speech analysis is Negative")


# Looks correct!
