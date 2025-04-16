from ultralytics import YOLO
import cv2

input_source = 'videos/IMG_8314.mp4'
cap = cv2.VideoCapture(input_source)
model = YOLO('yolomodels/yolo11n.pt')

cv2.namedWindow('People on Bridge', cv2.WINDOW_NORMAL)
cv2.resizeWindow('People on Bridge', 800, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    results = model.track(source=frame, persist=True)
    for result in results:
        processed_frame = result.plot()
        cv2.imshow('People on Bridge', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()