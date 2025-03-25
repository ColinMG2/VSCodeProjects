from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolomodels/best.pt')

# Results
input_source = input('Enter the path to the video: ')
results = model.predict(source=input_source, stream=True)

cv2.namedWindow('results', cv2.WINDOW_NORMAL)
cv2.resizeWindow('results', 800, 600)

# Loop through the results and display them
for result in results:
    frame = result.plot()  # Get the frame with predictions drawn
    cv2.imshow('results', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cv2.destroyAllWindows()