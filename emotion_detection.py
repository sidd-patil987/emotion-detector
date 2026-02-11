import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load model
model = load_model("emotion_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emoji mapping
emotion_emoji = {
    'Angry': '😡',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Sad': '😢',
    'Surprise': '😲',
    'Neutral': '😐'
}

# Smoothing
emotion_history = deque(maxlen=7)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion_history.append(emotion_index)

        final_index = max(set(emotion_history), key=emotion_history.count)
        emotion = emotion_labels[final_index]
        confidence = np.max(prediction) * 100

        emoji = emotion_emoji[emotion]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame,
                    f"{emoji} {emotion} {confidence:.1f}%",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2)

    cv2.imshow("Emotion Detector 😃", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
