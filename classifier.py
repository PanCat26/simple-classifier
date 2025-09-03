import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO

yolo_model = YOLO("yolov8n-seg.pt")
owner_image = face_recognition.load_image_file("owner.jpg")
owner_face_encoding = face_recognition.face_encodings(owner_image)[0]
cap = cv2.VideoCapture(0)

def classify_frame(frame):
    label = "Nobody"

    results = yolo_model(frame)[0]

    if results.masks is not None:
        for mask in results.masks.data:
            mask_np = mask.cpu().numpy()
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 0] = (mask_np * 255).astype(np.uint8)  # blue
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for encoding in face_encodings:
        if face_recognition.compare_faces([owner_face_encoding], encoding)[0]:
            return "Owner", frame
        else:
            label = "Other person"

    for box in results.boxes.data:
        cls = int(box[-1])
        if cls == 0:
            label = "Other person"
        elif cls in [15, 16]:
            return "Pet", frame

    if not face_encodings and len(results.boxes) == 0:
        label = "Nobody"

    return label, frame


while True:
    successful, frame = cap.read()
    if not successful:
        print("Failed to capture frame. Exiting...")
        break

    small_frame = cv2.resize(frame, (640, 480))
    detected_label, frame = classify_frame(small_frame)

    cv2.putText(frame, f"Detected: {detected_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Classification - Press 'q' to Exit", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
