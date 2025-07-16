import cv2
import numpy as np
from tensorflow.keras.models import load_model
from playsound import playsound
import threading

'''
Note: Your hair could be blocking the eyes, for the detection to work properly it is reccomended to tie your hair back
'''


# Load trained CNN model
model = load_model('eye_state_model.h5')

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open webcam
cap = cv2.VideoCapture(0)

score = 0  # To track drowsiness
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        closed_count = 0

        for (ex, ey, ew, eh) in eyes[:2]:  # Look at max 2 eyes
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (24, 24))
            eye_img = eye_img / 255.0
            eye_img = eye_img.reshape(1, 24, 24, 1)

            prediction = model.predict(eye_img, verbose=0)
            if prediction < 0.5:
                closed_count += 1

        # If both eyes are closed
        if closed_count == 2:
            score += 1
            cv2.putText(frame, "DROWSY!", (10, 50), font, 1, (0, 0, 255), 2)
        else:
            score = max(score - 1, 0)

        # If drowsy for too long
        if score > 5:
            cv2.putText(frame, "ALERT! WAKE UP!", (100, 100), font, 1.2, (0, 0, 255), 3)
            threading.Thread(target=playsound, args=('alarm.mp3',), daemon=True).start()

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
