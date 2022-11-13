# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT
import sys


"""Simple test for using adafruit_motorkit with a stepper motor"""
import time
import board
from adafruit_motorkit import MotorKit

kit = MotorKit(i2c=board.I2C())

args = sys.argv

print(args)

for i in range(int(float(args[-1])*100)):
    kit.stepper1.onestep()
    time.sleep(0.01)
