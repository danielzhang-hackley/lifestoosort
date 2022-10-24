print('\033c')

import numpy as np
import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    grayscale = cv2.cvtColor(cv2.blur(frame, (5, 5)), cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    cv2.imshow('thresh', thresh)
    cv2.imshow('frame', frame)

    print("\rwhats up", end='')

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
