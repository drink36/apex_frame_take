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
input_movie = cv2.VideoCapture("test3.mp4")
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


def get_cell_locations(lmks):
    def get_rect(center, width):
        point1 = np.array(center) - int(width / 2)
        point2 = np.array(center) + int(width / 2)
        return tuple(point1), tuple(point2)

    cells = {}
    cell_width = int((lmks['top_lip'][6][0] - lmks['top_lip'][0][0]) / 2)

    key = 'top_lip'
    points = np.array(lmks[key])
    left_lip_rect = get_rect(points[0], cell_width)
    right_lip_rect = get_rect(points[6], cell_width)
    cells['left_lip'] = left_lip_rect
    cells['right_lip'] = right_lip_rect

    key = 'chin'
    point = lmks[key][int(len(lmks[key]) / 2)]
    rect_point1 = (point[0] - int(cell_width / 2), point[1] - cell_width)
    rect_point2 = (point[0] + int(cell_width / 2), point[1])
    chin_rect = (rect_point1, rect_point2)
    cells['chin_rect'] = chin_rect

    key = 'nose_tip'
    point = lmks[key][0]
    left_nose_rect_point1 = (
        point[0] - cell_width, left_lip_rect[0][1] - cell_width)
    left_nose_rect_point2 = (point[0], left_lip_rect[0][1])
    left_nose_rect = (left_nose_rect_point1, left_nose_rect_point2)
    cells['left_nose'] = left_nose_rect

    point = lmks[key][4]
    right_nose_rect_point1 = (point[0], right_lip_rect[0][1] - cell_width)
    right_nose_rect_point2 = (point[0] + cell_width, right_lip_rect[0][1])
    right_nose_rect = (right_nose_rect_point1, right_nose_rect_point2)
    cells['right_nose'] = right_nose_rect

    key = 'left_eye'
    point = lmks[key][0]
    left_eye_rect_point1 = (point[0] - cell_width,
                            int(point[1] - cell_width / 2))
    left_eye_rect_point2 = (point[0], int(point[1] + cell_width / 2))
    left_eye_rect = (left_eye_rect_point1, left_eye_rect_point2)
    cells['left_eye'] = left_eye_rect

    key = 'right_eye'
    point = lmks[key][3]
    right_eye_rect_point1 = (point[0], int(point[1] - cell_width / 2))
    right_eye_rect_point2 = (
        point[0] + cell_width, int(point[1] + cell_width / 2))
    right_eye_rect = (right_eye_rect_point1, right_eye_rect_point2)
    cells['right_eye'] = right_eye_rect

    left_point = lmks['left_eyebrow'][2]
    right_point = lmks['right_eyebrow'][2]
    center_point = (int((left_point[0] + right_point[0]) / 2),
                    int((left_point[1] + right_point[1]) / 2))

    center_eyebrow_rect = get_rect(center_point, cell_width)
    cells['center_eyebrow'] = center_eyebrow_rect

    left_rect_point1 = (int(center_point[0] - cell_width * 3 / 2),
                        int(center_point[1] - cell_width / 2))
    left_rect_point2 = (int(center_point[0] - cell_width * 1 / 2),
                        int(center_point[1] + cell_width / 2))
    left_eyebrow_rect = (left_rect_point1, left_rect_point2)
    cells['left_eyebrow'] = left_eyebrow_rect

    right_rect_point1 = (int(center_point[0] + cell_width * 1 / 2),
                         int(center_point[1] - cell_width / 2))
    right_rect_point2 = (int(center_point[0] + cell_width * 3 / 2),
                         int(center_point[1] + cell_width / 2))
    right_eyebrow_rect = (right_rect_point1, right_rect_point2)
    cells['right_eyebrow'] = right_eyebrow_rect

    return cells, cell_width


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
    cell_locations, cell_width = get_cell_locations(face_landmarks_list[0])
    for cell in cell_locations.values():
        cv2.rectangle(frame, (cell[0][0], cell[1][1]),
                      (cell[1][0], cell[0][1]), (0, 0, 255), 1)

    # for face_landmarks in face_landmarks_list:
    #     for facial_feature in face_landmarks.keys():
    #         point = face_landmarks[facial_feature]
    #         for i in range(len(point)-1):
    #             cv2.line(frame, point[i], point[i+1], (0, 0, 255), 1)
    # # Label the results
    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)


# All done!
input_movie.release()
cv2.destroyAllWindows()
