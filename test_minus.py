import cv2
import face_recognition
import numpy as np


def minus(img1, img2):
    diff = cv2.subtract(img1, img2)
    return diff


def minus2(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return diff


def minus3(img1, img2):
    diff = img1-img2
    return diff


# python main func
if __name__ == '__main__':
    # read image
    img1 = cv2.imread('onset.jpg')
    img2 = cv2.imread('apex.jpg')
    # minus
    diff = minus(img2, img1)
    diff2 = minus2(img2, img1)
    diff3 = minus3(img2, img1)
    # save image
    cv2.imwrite('diff1.jpg', diff)
    cv2.imwrite('diff2.jpg', diff2)
    cv2.imwrite('diff3.jpg', diff3)
