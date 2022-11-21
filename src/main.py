import cv2
from adafruit_servokit import ServoKit
from adafruit_motorkit import MotorKit
import board
from utils import screw_bolt_other, move_output_bin, int_string_format


if __name__ == "__main__":
    output_kit = ServoKit(channels=16)
    belt_kit = MotorKit(i2c=board.I2C())

    cap = cv2.VideoCapture('/dev/video0')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    i = 1
    classifications = {"screw": 0, "bolt": 0, "other": 0}
    while True:
        ret, img = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if i % 125 == 0:
            classifications = {"screw": 0, "bolt": 0, "other": 0}
            i = 1

        try:
            fastener_type, ratio, sketches, thresh, head = screw_bolt_other(img)
            classifications[fastener_type] += 1
            # move_output_bin(fastener_type)

            # USE THE BELOW VALUE TO DETERMINE OUTPUT RAMP ANGLE
            most_likely_fastener_type = max(classifications, key=classifications.get)
            print("\r", int_string_format(i), most_likely_fastener_type, ratio, end='')

            cv2.imshow('sketches', sketches)
            cv2.imshow('thresholds', thresh)
            cv2.imshow('edges', head)
        except:
            pass

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
