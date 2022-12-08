import cv2
from adafruit_servokit import ServoKit
from adafruit_motorkit import MotorKit
import board
from utils import classify_fastener, move_output_bin, int_string_format, move_belt
import time
import math


class Fastener:
    def __init__(self, klass, dist):
        self._type = klass
        self._dist = dist

    def get_type(self):
        return str(self._type)

    def set_type(self, klass):
        self._type = str(klass)

    def get_dist(self):
        return float(self._dist)

    def set_dist(self, dist):
        self._dist = dist

    def change_dist(self, dist):
        self._dist += dist


class LoadedBelt:
    def __init__(self, fastener_list=None, radius=17., length=100, output_kit=ServoKit(channels=16), belt_kit=MotorKit(i2c=board.I2C())):
        # fasteners is a list of Fasteners
        self._fastener_list = [] if fastener_list is None else fastener_list
        self._radius = radius  # millimeters radius of pipe
        self._length = length  # millimeters length of belt (fov to end)
        self._next_type = self._fastener_list[0].get_type()
        self._next_dist = self._fastener_list[0].get_dist()

        self._output_kit = output_kit
        self._belt_kit = belt_kit


    def deg_to_dist(self, deg):
        return self._radius * (deg * math.pi / 180)

    def dist_to_deg(self, dist):
        return dist / self._radius * (180 / math.pi)

    def push(self, fastener):
        self._fastener_list.append(fastener)

    def pop(self):
        return self._fastener_list.pop(0)

    def rotate(self):
        first_fastener_type = self._fastener_list[0].get_type()
        first_fastener_dist = self._fastener_list[0].get_dist()
        move_output_bin(self._output_kit, first_fastener_type)
        move_belt(self._belt_kit, self.dist_to_deg(first_fastener_dist))

        for fastener in self._fastener_list:
            fastener.change_dist(first_fastener_dist)
        self.pop()


if __name__ == "__main__":
    output_loc = "other"
    belt_move_deg = 180
    fasteners = []

    output_kit = ServoKit(channels=16)
    belt_kit = MotorKit(i2c=board.I2C())
    cap = cv2.VideoCapture('/dev/video0')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    i = 1
    classifications = {"non-hex": 0, "hex": 0, "other": 0}
    while True:
        ret, img = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if i % 50 == 0:

            most_likely_fastener_type = max(classifications, key=classifications.get)
            start_loc = 360

            """
            fasteners.append(Fastener(most_likely_fastener_type, start_loc))

            #increment the locations of the fasteners to what they will be after the belt is moved
            kill_list = []
            for fastener_index in range(len(fasteners)):
                fasteners[fastener_index]._deg -= belt_move_deg
                if fasteners[fastener_index]._deg <= 0:
                    fastener_at_end = fasteners[fastener_index]
                    output_loc = fastener_at_end._id
                    kill_list.append(fastener_index)
                    print("current output: " + output_loc)
            kill_list.reverse()
            for k in kill_list:
                fasteners.pop(k)

            # move the output bin
            move_output_bin(output_kit, output_loc)
            # let the output servo catch up to the belt
            time.sleep(2)
            # move the belt
            # move_belt(belt_kit, belt_move_deg)
            """



            classifications = {"non-hex": 0, "hex": 0, "other": 0}
            i = 1

        crop_distance = img.shape[0] // 15
        img = img[crop_distance: img.shape[0] - crop_distance, :]
        try:
            fastener_type, ratio, sketches, thresh, head = classify_fastener(img)
            if i >= 10:
                classifications[fastener_type] += 1
            # move_output_bin(fastener_type)

            # USE THE BELOW VALUE TO DETERMINE OUTPUT RAMP ANGLE
            most_likely_fastener_type = max(classifications, key=classifications.get)
            print("\r", int_string_format(i), most_likely_fastener_type, ratio, end='')

            cv2.imshow('original', img)
            cv2.imshow('sketches', sketches)
            cv2.imshow('thresholds', thresh)
            cv2.imshow('edges', head)
        except:
            pass

        if cv2.waitKey(1) == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()
