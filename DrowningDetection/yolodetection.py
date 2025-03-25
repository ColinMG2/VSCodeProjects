from ultralytics import YOLO
import cv2
from multiprocessing import Process

def run_model(model_path, window_name, input_source):
    # Load model
    model = YOLO(model_path)

    # Results
    results = model.predict(source=input_source, stream=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # Loop through the results and display them
    for result in results:
        frame = result.plot()  # Get the frame with predictions drawn
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_source = input("Enter the path to the video file: ")

    # Create a process for each model
    process1 = Process(target=run_model, args=('yolomodels/best.pt', 'results_best', input_source))
    process2 = Process(target=run_model, args=('yolomodels/yolo11m-pose.pt', 'results_pose', input_source))

    # Start the processes
    process1.start()
    process2.start()

    # Wait for the processes to finish
    process1.join()
    process2.join()