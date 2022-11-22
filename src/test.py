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


