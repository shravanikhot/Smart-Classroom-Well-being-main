import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import logging

# Setup logging
logging.basicConfig(filename="errors.log", level=logging.ERROR)

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained mental health model
mental_health_model = tf.keras.models.load_model("trained_mobilenetv2_model.h5")

# Start video capture
cap = cv2.VideoCapture(0)

# Sampling and statistics setup
frame_counter = 0
sample_interval = 10  # Process every 10th frame
emotion_data = {"angry": [], "disgust": [], "fear": [], "happy": [], "sad": [], "surprise": [], "neutral": []}
dominant_emotions = []
combined_predictions = []  # Weighted combination of DeepFace and mental health model
positive_count = 0
negative_count = 0

# Store the last prediction from the combined model
last_combined_prediction = None  # 'Positive' or 'Negative'


def preprocess_face(face_img):
    """Preprocess the face image for the mental health model."""
    resized_face = cv2.resize(face_img, (224, 224))
    return np.expand_dims(resized_face, axis=0) / 255.0


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Increment frame counter
    frame_counter += 1

    # Convert frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face detected
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face region for emotion and mental health analysis
        face_img = frame[y:y + h, x:x + w]

        try:
            # DeepFace emotion analysis
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

            # Handle output format based on DeepFace version
            if isinstance(analysis, list):
                emotion_scores = analysis[0]['emotion']
                dominant_emotion = analysis[0]['dominant_emotion']
            else:
                emotion_scores = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']

            # Record probabilities for each emotion
            for emotion, score in emotion_scores.items():
                emotion_data[emotion.lower()].append(score)

            # Record the dominant emotion for this frame
            dominant_emotions.append(dominant_emotion)

            # Display the dominant emotion above the face rectangle
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

            # Mental health model prediction (every 10th frame)
            if frame_counter % sample_interval == 0:
                normalized_face = preprocess_face(face_img)
                prediction = mental_health_model.predict(normalized_face)
                class_label = np.argmax(prediction[0]) + 1  # Class 1/2
                confidence = np.max(prediction[0])

                # Weightage: 70% DeepFace, 30% mental health model
                deepface_positive = dominant_emotion in ['happy', 'neutral']
                mental_health_positive = class_label == 1
                combined_positive_score = 0.7 * deepface_positive + 0.3 * mental_health_positive

                # Final prediction based on weighted score
                combined_prediction = 'Positive' if combined_positive_score >= 0.5 else 'Negative'
                combined_predictions.append(combined_prediction)

                if combined_prediction == 'Positive':
                    positive_count += 1
                else:
                    negative_count += 1

                last_combined_prediction = combined_prediction

        except Exception as e:
            logging.error(f"Error in analysis at frame {frame_counter}: {e}")

        # Display the last prediction continuously
        if last_combined_prediction:
            cv2.putText(frame, f"Well-Being: {last_combined_prediction}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Emotion & Mental Health Analysis', frame)

    # Break loop on 'ESC' key press
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Data aggregation and analysis
standard_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_means = {emotion: np.mean(emotion_data.get(emotion, [0])) for emotion in standard_emotions}
dominant_emotion_counts = Counter(dominant_emotions)
combined_counts = Counter(combined_predictions)

# Well-Being Analysis
def analyze_wellbeing(dominant_emotions):
    transitions = sum(1 for i in range(1, len(dominant_emotions)) if dominant_emotions[i] != dominant_emotions[i - 1])
    predominant_negative = sum(dominant_emotions.count(emotion) for emotion in ['angry', 'sad', 'fear', 'disgust'])
    predominant_positive = dominant_emotions.count('happy')
    neutral_count = dominant_emotions.count('neutral')

    # Emotional Stability Analysis
    stability = "Stable" if transitions / len(dominant_emotions) < 0.3 else "Unstable"

    # Risk Analysis
    depression_risk = "High" if dominant_emotions.count('sad') / len(dominant_emotions) > 0.3 else "Low"
    anxiety_risk = "High" if dominant_emotions.count('fear') / len(dominant_emotions) > 0.2 else "Low"
    stress_risk = "High" if transitions / len(dominant_emotions) > 0.4 or predominant_negative > predominant_positive else "Low"

    return {
        "Emotional Stability": stability,
        "Risk of Depression": depression_risk,
        "Risk of Anxiety": anxiety_risk,
        "Risk of Stress": stress_risk,
    }

wellbeing_report = analyze_wellbeing(dominant_emotions)

# Display Plots
def display_mean_probabilities(emotion_means):
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_means.keys(), emotion_means.values(), color='skyblue')
    plt.title('Mean Emotion Probabilities')
    plt.ylabel('Probability')
    plt.xlabel('Emotions')
    plt.show()

def display_combined_predictions(combined_counts):
    plt.figure(figsize=(8, 8))
    plt.pie(combined_counts.values(), labels=combined_counts.keys(), autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
    plt.title('Combined Well-Being Predictions')
    plt.show()

def display_dominant_emotions(dominant_emotion_counts):
    plt.figure(figsize=(8, 8))
    colors = plt.cm.Paired(range(len(dominant_emotion_counts)))
    plt.pie(dominant_emotion_counts.values(), labels=dominant_emotion_counts.keys(), autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Dominant Emotion Distribution')
    plt.show()

def display_wellbeing_report(wellbeing_report):
    report_text = "\n".join([f"{key}: {value}" for key, value in wellbeing_report.items()])
    plt.figure(figsize=(10, 4))
    plt.text(0.1, 0.5, report_text, fontsize=12, wrap=True)
    plt.title('Well-Being Report')
    plt.axis('off')
    plt.show()
   

# Call visualization functions
display_mean_probabilities(emotion_means)
display_combined_predictions(combined_counts)
display_dominant_emotions(dominant_emotion_counts)
display_wellbeing_report(wellbeing_report)
