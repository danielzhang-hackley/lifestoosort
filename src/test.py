<<<<<<< HEAD
import cv2
cap = cv2.VideoCapture('/dev/video0')
ret, frame = cap.read()

while(True):
    cv2.imshow("AAH", frame)
    if cv2.waitKey(1) & 0xFF == ord('y'):
        cv2.imwrite('./images/bolt_real.png', frame)
        cv2.destroyAllWindows()
        break

cap.release()


=======
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
>>>>>>> b2993514062885104cb41f47e3ed808332f55039
