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
"""
print('\033c')

import cv2
import numpy as np


# PROCESS IMAGE AND CREATE CONTOURS
img = cv2.imread(r'./images/wood_screw.jpg')
grayscale = cv2.cvtColor(cv2.blur(img, (10, 10)), cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(grayscale, 230, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
max_area = cv2.contourArea(cnt)
for cont in contours:
    if cv2.contourArea(cont) > max_area:
        cnt = cont
        max_area = cv2.contourArea(cont)

perimeter = cv2.arcLength(cnt, True)
epsilon = 0.005*cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)


# CREATE CONVEX HULL
convex_hull_points = cv2.convexHull(approx)
convex_hull = cv2.convexHull(approx, returnPoints=False)
convexity_defects = cv2.convexityDefects(approx, convex_hull)


# FIND CONVEXITY DEFECTS
# find the indices of convexity defect where the distance between hull and approximation is greatest
p1_idx = [0, 0]  # the index in convexity_defects of the max, max
p2_idx = [0, 0]
for i, defect in enumerate(convexity_defects):
    defect = defect[0]
    if defect[3] > p1_idx[1]:
        p2_idx = p1_idx
        p1_idx = [i, defect[3]]
    elif defect[3] > p2_idx[1]:
        p2_idx = [i, defect[3]]

# find the indices of the second closest pair of points on the convex hull
closest = [(0, -1), (0, -1), float('inf')]   # (1st index in approx which is element of convexity_defects, upper or lower),
                                             # (2nd index, upper lower), min
second_closest = [(0, -1), (0, -1), float('inf')]
for i, first_point_idx in enumerate(convexity_defects[p1_idx[0]][0][0:2]):
    for j, second_point_idx in enumerate(convexity_defects[p2_idx[0]][0][0:2]):
        first_point = approx[first_point_idx][0]
        second_point = approx[second_point_idx][0]

        difference = second_point - first_point
        distance = np.linalg.norm(difference)

        if distance < closest[-1]:
            second_closest = closest
            closest = [(first_point_idx, i), (second_point_idx, j), distance]
        elif distance < second_closest[-1]:
            second_closest = [(first_point_idx, i), (second_point_idx, j), distance]


# FIND NOTCHES
# if lower bound, we can move forward one, if upper bound, we can move bacward one
dct = {0: 1, 1: -1}
notch1_idx = second_closest[0][0] + dct[second_closest[0][1]]
notch2_idx = second_closest[1][0] + dct[second_closest[1][1]]
notch1 = approx[notch1_idx][0]
notch2 = approx[notch2_idx][0]


# FIND BOUNDING BOX
approx = np.vstack([approx[: min(notch1_idx + 1, notch2_idx + 1)], approx[max(notch1_idx, notch2_idx):]])

rect = cv2.minAreaRect(approx)
box = cv2.boxPoints(rect)
box = np.int0(box)


# FIND AREAS AND COMPARE
box_area = cv2.contourArea(box)
head_area = cv2.contourArea(approx)
ratio = head_area / box_area
if ratio >= .75:
    print("hex bolt: ", ratio)
else:
    print("other screw/bolt: ", ratio)


# DRAW EVERYTHING
cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)
cv2.drawContours(img, [convex_hull_points], -1, (255, 0, 0), 2)
img = cv2.circle(img, notch1, radius=5, color=(0, 255, 0), thickness=-1)
img = cv2.circle(img, notch2, radius=5, color=(0, 255, 0), thickness=-1)
cv2.drawContours(img,[box],0,(255, 0, 255),2)



# HANDLE WINDOWS
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



