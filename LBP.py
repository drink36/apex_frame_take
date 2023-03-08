from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io, data_dir, filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import face_recognition
import cv2


def face_taker(frame):
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    # Loop through each face in this frame of video
    crop_img = frame[face_locations[0][0]:face_locations[0]
                     [2], face_locations[0][3]:face_locations[0][1]]
    # save face photo
    return crop_img


# settings for LBP
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数
# 读取图像
image = cv2.imread('onset.jpg')
image = face_taker(image)
# 显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# gray image
image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
plt.subplot(231)
plt.imshow(image, plt.cm.gray)
# LBP process
lbp = local_binary_pattern(image, n_points, radius)
plt.subplot(232)
plt.imshow(lbp, plt.cm.gray)
n_image = cv2.imread('apex.jpg')
n_image = face_taker(n_image)
# 显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
n_image1 = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)
# gray image
n_image = cv2.cvtColor(n_image1, cv2.COLOR_BGR2GRAY)
plt.subplot(233)
plt.imshow(n_image, plt.cm.gray)
# LBP process
n_lbp = local_binary_pattern(n_image, n_points, radius)
plt.subplot(234)
plt.imshow(n_lbp, plt.cm.gray)
# calculate the difference between two images(with LBP)
diff = lbp - n_lbp
cv2.imwrite('diff_re.jpg', diff)
plt.subplot(235)
plt.imshow(diff, plt.cm.gray)
plt.savefig('diff.jpg')
plt.show()
