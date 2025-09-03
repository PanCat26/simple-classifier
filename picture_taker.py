import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Capture Face - Press 's' to Save", frame)
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite("owner.jpg", frame)
        break
cap.release()
cv2.destroyAllWindows()
