import utils
import numpy as np

print('\033c')

"""
[[[383 163]]

 [[382 164]]

 [[374 164]]

 [[373 165]]]

"""

x = np.array([1, 2, 3])
y = np.array([[1],
              [2],
              [3]])

print(utils.add_affine(x, axis=1))
print(utils.add_affine(y, axis=1))
