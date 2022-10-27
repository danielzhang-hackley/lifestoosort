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
from polygons_static_4 import screw_bolt_other


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    fastener_type, ratio, sketches, thresh, head = screw_bolt_other(img)
    print("\r", fastener_type, ratio, end='')

    # HANDLE WINDOWS
    cv2.imshow('sketches', sketches)
    cv2.imshow('thresholds', thresh)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



