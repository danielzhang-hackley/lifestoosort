print('\033c')

import numpy as np
import cv2


img = cv2.imread(r'./images/wood_screw.jpg')

grayscale = cv2.cvtColor(cv2.blur(img, (10, 10)), cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grayscale, 210, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cnt = contours[0]
max_area = cv2.contourArea(cnt)

for cont in contours:
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)

perimeter = cv2.arcLength(cnt, True)
epsilon = 0.0125*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# print("\r" + str(len(cnt)), end='')
cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)


print(len(approx))

cv2.imshow('img', img)
cv2.imshow('thresh', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()


