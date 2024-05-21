import cv2

cap = cv2.VideoCapture(0)

has_frame, img = cap.read()
if not has_frame:
    raise ValueError('Invalid camera')

while True:
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()