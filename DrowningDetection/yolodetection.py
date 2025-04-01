from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from threading import Thread
from ultralytics.engine.results import Keypoints

def run_model(model_path, window_name, input_source, is_pose=False):
    # Load model
    model = YOLO(model_path)

    # Results
    if is_pose==False:
        results = model.predict(source=input_source, stream=True)
    else:
        results = model.predict(source=input_source, stream=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # Loop through the results and display them
    for result in results:
        frame = result.plot()  # Get the frame with predictions drawn
        cv2.imshow(window_name, frame)

        if is_pose==True and result.keypoints is not None:
            base_tensor = result.keypoints.data
            keypoints = Keypoints(base_tensor, orig_shape=frame.shape)
            normalized_keypoints = np.squeeze(keypoints.xyn.cpu().numpy())
            x_norm, y_norm = normalized_keypoints[:, 0], normalized_keypoints[:, 1]
            keypoints_data = {'keypoints': ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'] ,
                                        'x': x_norm,
                                        'y': y_norm}
            df = pd.DataFrame(keypoints_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_source = input("Enter the path to the video file: ")

    # Create a process for each model
    thread1 = Thread(target=run_model, args=('yolomodels/best.pt', 'results_best', input_source, False))
    thread2 = Thread(target=run_model, args=('yolomodels/yolo11n-pose.pt', 'results_pose', input_source, True))

    # Start the processes
    thread1.start()
    thread2.start()

    # Wait for the processes to finish
    thread1.join()
    thread2.join()