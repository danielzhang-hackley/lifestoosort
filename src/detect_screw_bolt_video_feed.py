import cv2
from detect_screw_bolt import screw_bolt_other
from move_output_bin import move_output_bin

cap = cv2.VideoCapture('/dev/video0')

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    try:
        fastener_type, ratio, sketches, thresh, head = screw_bolt_other(img)
        # move_output_bin(fastener_type)
        print("\r", fastener_type, ratio, end='')

        # HANDLE WINDOWS
        cv2.imshow('sketches', sketches)
        cv2.imshow('thresholds', thresh)
        cv2.imshow('edges', head)
    except:
        pass

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



