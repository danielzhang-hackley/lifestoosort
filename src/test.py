print('\033c')

import cv2
import numpy as np


def rotate_image(image, angle, coord):
    rot_mat = cv2.getRotationMatrix2D(coord, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# PROCESS IMAGE AND CREATE CONTOURS
img = cv2.imread(r'./images/wood_screw.jpg')

img = rotate_image(img, 45, (10, 10))
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()