import cv2
from detect_screw_bolt import screw_bolt_other
from move_output_bin import move_output_bin

cap = cv2.VideoCapture('/dev/video0')
cap.set(cv2.CAP_PROP_EXPOSURE, -8)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, img = cap.read()
    img_height, img_width = img.shape[0], img.shape[1]
    h_start = int(0.15 * img_height)
    h_end = int(img_height - 0.15 * img_height)
    w_start = int(0.15 * img_width)
    w_end = int(img_width - 0.15 * img_width)
    img = img[h_start: h_end, w_start: w_end]
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    try:
        fastener_type, ratio, sketches, thresh, head = screw_bolt_other(img, light_threshold=160)
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



