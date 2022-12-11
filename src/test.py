import numpy as np
import cv2 as cv

class Fastener:
    def __init__(self, klass, dist):
        self._type = klass
        self._dist = dist

    def get_type(self):
        return str(self._type)

    def set_type(self, klass):
        self._type = str(klass)

    def get_dist(self):
        return float(self._dist)

    def set_dist(self, dist):
        self._dist = dist

    def increase_dist(self, dist):
        self._dist += dist

class LoadedBelt:
    diameter = 34.  # millimeters
    radius = diameter/2

    def __init__(self, fastener_list=None):
        # fasteners is a list of Fasteners
        self._fastener_list = [] if fastener_list is None else fastener_list

    def rotate(self, deg):
        [fastener.increase_dist(deg) for fastener in self._fastener_list]

    def get_pos(self):
        [print(fastener.get_dist()) for fastener in self._fastener_list]


'''
img = cv.imread(r"./images/bolt_real_rotated.png")
mask = np.zeros(img.shape, np.uint8)
grayscale = cv.cvtColor(cv.blur(img, (2, 2)), cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(grayscale, 148, 255, cv.THRESH_BINARY)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

max_area = 0
cnt = contours[0]
for cont in contours:
    if cv.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv.contourArea(cont)
        cimg = np.zeros_like(thresh)

        print(max_area)

print("***************")
cv.drawContours(mask, contours, -1, (0,255,0),1)
print(max_area)
print(img.shape[0] * img.shape[1])
print(max_area / (img.shape[0] * img.shape[1]))
cv.imshow("test", mask)
cv.waitKey(0)
cv.destroyAllWindows()
'''


x = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
print(np.sum(x == np.array([1, 2, 3])))