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

    ret, thresh = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    try:
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)

        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        print("\r" + str(len(cnt)), end='')
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)
    except:
        pass


    cv2.imshow('frame', frame)
    cv2.imshow('thresh', thresh)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print(cnt)
