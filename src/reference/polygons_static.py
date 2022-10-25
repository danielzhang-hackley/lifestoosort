"""
New strategy:
Polygonal approximation of the whole fastener. If there are no concave points, disregard
If there are exactly two concave points, then it is either a screw or bolt.
Connect the concave points with a line. Draw two more parallel lines the same distance from that intersecting line.
For these two new lines, record where they intersect the polygonal estimation.
Focus on the side with the greater area after these two lines intersect
Create a rectangular bounding box for the big area
Calculate the difference in area between the contour estimation and the bounding box estimation. if it is below _%
then it is a bolt, otherwise it is a screw.


A point is concave, then it is not on the convex hull


for contour in contours:
    convexHull = cv2.convexHull(contour)
    cv2.drawContours(image, [convexHull], -1, (255, 0, 0), 2)


convexityDefects = cv2.convexityDefects(contour, convexhull)


"""



print('\033c')

import cv2


img = cv2.imread(r'./images/hex_bolt_zoom.jpg')

grayscale = cv2.cvtColor(cv2.blur(img, (5, 5)), cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grayscale, 230, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cnt = contours[0]
max_area = cv2.contourArea(cnt)

for cont in contours:
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)

perimeter = cv2.arcLength(cnt, True)
epsilon = 0.0175*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# print("\r" + str(len(cnt)), end='')
cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)


print(len(contours))
print(len(approx))

cv2.imshow('img', img)
cv2.imshow('thresh', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()


