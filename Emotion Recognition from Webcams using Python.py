# By: Tim Tarver

# Emotion Recognition through Webcams using Python

import cv2
from deepface import DeepFace
import numpy

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier() # Process the facial recognition

# In case the the picture does not load, print an error

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    
    print("Error Loading XML File")

# The line below requisites the input picture from the webcam

video = cv2.VideoCapture(0)

# Now, we will loop through the picture to check if we are obtaining
# the video feed we need and uses it.

while video.isOpened():

    _, frame = video.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Change video to GreyScale
    face_recognition = face_cascade.detectMultiScale(grey, scaleFactor=1.1,
                                                     minNeighbors=5)

    for x, y, width, height in face_recognition:

        image = cv2.rectangle(frame, (x, y),
                              (x+width, y+height), (0, 0, 255), 1)

        try:
            analyze_face = DeepFace.analyze(frame, actions=['emotion'])
            print(analyze['dominant_emotion'])

        except:
            print("No Face")

        cv2.imshow('video', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    video.release()    
