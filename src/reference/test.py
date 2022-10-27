print('\033c')

import cv2
import numpy as np

x = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(np.flip(x, axis=1))