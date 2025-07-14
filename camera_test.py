# camera_test.py
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Test Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
