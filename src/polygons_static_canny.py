print('\033c')

import numpy as np
import cv2


img = cv2.imread(r'./images/hex_bolt_zoom.jpg')

t_lower = 800  # Lower Threshold
t_upper = 900  # Upper threshold
aperture_size = 5  # Aperture size

# Applying the Canny Edge filter
# with Custom Aperture Size
edge = cv2.Canny(cv2.blur(img, (10, 10)), t_lower, t_upper,
                 apertureSize=aperture_size)
cv2.imshow('original', img)
cv2.imshow('edge', edge)

contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

for cont in contours:
    cnt = cont
    max_area = cv2.contourArea(cont)

    perimeter = cv2.arcLength(cnt, True)
    epsilon = 0.01*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    polygons_edge = cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)
cv2.imshow('polygons from edge', polygons_edge)

print(len(contours))
print(len(approx))

cv2.waitKey(0)
cv2.destroyAllWindows()


