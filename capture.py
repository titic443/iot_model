import cv2

cap = cv2.VideoCapture(33)
ret, frame = cap.read()
if ret:
    cv2.imwrite("test_face.jpg", frame)
    print("saved: test_face.jpg")
else:
    print("failed: cannot capture from video0")
cap.release()
