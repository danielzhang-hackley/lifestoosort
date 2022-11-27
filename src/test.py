import utils
import numpy as np

print('\033c')

"""
[[[383 163]]

 [[382 164]]

 [[374 164]]

 [[373 165]]]

"""

x = np.array([[[1, 0]], [[2, 2]]])
print(utils.least_squares_perp_offset(x))