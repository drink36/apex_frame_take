import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt


# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("test2.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 60, (640, 360))
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_landmarks_list = []
frame_number = 0
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    # Quit when the input video file ends
    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_landmarks_list = face_recognition.face_landmarks(frame)

    # Label the results
    for (top, right, bottom, left) in face_locations:
        for face_landmarks in face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                point = face_landmarks[facial_feature]
                for i in range(len(point)-1):
                    cv2.line(frame, point[i], point[i+1], (0, 0, 255), 1)
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
