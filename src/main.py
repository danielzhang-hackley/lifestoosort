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
    def __init__(self, fastener_list=None, radius=17., length=135, output_kit=ServoKit(channels=16), belt_kit=MotorKit(i2c=board.I2C())):
        # fasteners is a list of Fasteners
        self._fastener_list = [] if fastener_list is None else fastener_list
        self._radius = radius  # millimeters radius of pipe
        self._length = length  # millimeters length of belt (fov to end)

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
        if len(self._fastener_list) > 0:
            first_fastener_type = self._fastener_list[0].get_type()
            first_fastener_dist = self._fastener_list[0].get_dist()
        else:
            return None
        move_output_bin(self._output_kit, first_fastener_type)
        time.sleep(0.5)
        move_belt(self._belt_kit, int(self.dist_to_deg(self._length - first_fastener_dist) / 1.8))

        for fastener in self._fastener_list:
            fastener.change_dist(self._length - first_fastener_dist)
        self.pop()



def deg_to_dist(deg, radius):
    return radius * (deg * math.pi / 180)

def dist_to_deg(dist, radius):
    return dist / radius * (180 / math.pi)

def rotate(klass, dist, radius, length, belt_kit, output_kit):
    move_output_bin(output_kit, klass)
    time.sleep(0.5)
    move_belt(belt_kit, int(dist_to_deg(length - dist, radius) / 1.8))


if __name__ == "__main__":
    print("\033c")
    cap = cv2.VideoCapture('/dev/video0')
    output_kit=ServoKit(channels=16)
    belt_kit=MotorKit(i2c=board.I2C())
    radius = 17
    length = 135

    loaded_belt = LoadedBelt()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    first_iter = True
    i = 1
    classifications = {"non-hex": 0, "hex": 0, "other": 0}
    while True:
        ret, img = cap.read()
        if first_iter:
            fastener_type, ratio, sketches, thresh, head, dist = classify_fastener(img, light_threshold=144, blur=(2,2))
            first_iter = False
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if i % 40 == 0:
            most_likely_fastener_type = max(classifications, key=classifications.get)
            rotate(most_likely_fastener_type, dist, radius, length, belt_kit, output_kit)
            classifications = {"non-hex": 0, "hex": 0, "other": 0}
            i = 1
        
        top_crop_distance = int(img.shape[0] * 0.04)
        bottom_crop_distance = int(img.shape[0] * 0.11)
        img = img[top_crop_distance: img.shape[0] - bottom_crop_distance, :]
        try:
            fastener_type, ratio, sketches, thresh, head, dist = classify_fastener(img, light_threshold=144)
            if i >= 20:  # allow time for camera to focus
                classifications[fastener_type] += 1
            # move_output_bin(fastener_type)

            # USE THE BELOW VALUE TO DETERMINE OUTPUT RAMP ANGLE
            most_likely_fastener_type = max(classifications, key=classifications.get)
            print("\r", int_string_format(i) + " ", most_likely_fastener_type, dist, end='')

            cv2.imshow('original', img)
            cv2.imshow('sketches', sketches)
            cv2.imshow('thresholds', thresh)
            if head is not None:
                cv2.imshow('edges', head)
        except:
            pass

        if cv2.waitKey(1) == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()
