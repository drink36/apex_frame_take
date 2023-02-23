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

# Create an output movie file (make sure resolution/frame rate matches input video!)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter('output.avi', fourcc, 60, (640, 360))
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_landmarks_list = []
frame_number = 0


def own(filename):
    input_movie = cv2.VideoCapture(filename)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))-1
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, 1)
    rval, on = input_movie.read()
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, length-1)
    rval, off = input_movie.read()
    on = cv2.cvtColor(on, cv2.COLOR_BGR2GRAY)
    off = cv2.cvtColor(off, cv2.COLOR_BGR2GRAY)
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, 1)
    features = []
    frame_number = 0
    while True:
        ret, frame = input_movie.read()
        if not ret:
            break
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        # set rgb frame to gray
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        if(frame_number != 0):
            current_features = compute_cell_features(
                gray_frame, on, off, temp)
            feature = 0
            for key in current_features:
                feature += current_features[key]
            feature = feature / len(current_features)
            features.append(feature)
        print("Scanning frame {} / {}".format(frame_number, length-2))
        temp = gray_frame
        frame_number += 1
    padding = [0.0] * 1
    features = np.array(padding + features)
    apex_frame_idx = features.argmax()
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, apex_frame_idx)
    rval, frame = input_movie.read()
    cv2.imwrite('apex3.jpg', frame)
    return features, apex_frame_idx


def detect_lmks(frame):
    lmks = face_recognition.face_landmarks(frame)
    return lmks[0]


def get_cell(img, cell_location):
    point1, point2 = cell_location
    cell = img[point1[1]:point2[1], point1[0]:point2[0]]
    return cell


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


def compute_cell_difference(cell_t, cell_onset, cell_offset, cell_epsilon):
    numerator = (np.abs(cell_t - cell_onset) + 1.0)
    denominator = (np.abs(cell_t - cell_epsilon) + 1.0)
    difference = numerator / denominator

    numerator = (np.abs(cell_t - cell_offset) + 1.0)
    difference1 = numerator / denominator

    # difference = difference + difference1

    return difference.mean()


def compute_cell_features(frame_t, on_frame, off_frame, frame_epsilon):
    lmks = detect_lmks(frame_t)
    cell_locations, cell_width = get_cell_locations(lmks)
    cell_differences = {}
    frame_t = frame_t.astype(np.float32)
    on_frame = on_frame.astype(np.float32)
    off_frame = off_frame.astype(np.float32)
    frame_epsilon = frame_epsilon.astype(np.float32)

    for key in cell_locations:
        cell_location = cell_locations[key]
        cell_t = get_cell(frame_t, cell_location)
        cell_onset = get_cell(on_frame, cell_location)
        cell_offset = get_cell(off_frame, cell_location)
        cell_epsilon = get_cell(frame_epsilon, cell_location)

        cell_difference = compute_cell_difference(
            cell_t, cell_onset, cell_offset, cell_epsilon)
        cell_differences[key] = cell_difference
    return cell_differences


def draw_avg_plot(features, pred_apex_idx, clip_name):
    x = list(range(len(features)))
    plt.plot(x, features)
    plt.axvline(x=pred_apex_idx, label='pred apex idx at={}'.format(
        pred_apex_idx), c='red')
    plt.legend()
    plt.savefig('{}.png'.format(clip_name))
    plt.clf()
    plt.cla()
    plt.close()


features, apex_relative_idx = own("test3.mp4")
draw_avg_plot(features, apex_relative_idx, 'own3')
print(apex_relative_idx)
