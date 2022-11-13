from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

def move_output_bin(type):
	if (type == "screw"):
		kit.servo[1].angle = 40
	elif (type == "bolt"):
		kit.servo[1].angle = 140
	else:
		kit.servo[1].angle = 90
