from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from ultralytics.engine.results import Keypoints

def run_model(model_path, window_name, input_source):
    # Load model
    model = YOLO(model_path)

    # Results
    results = model.predict(source=input_source, stream=True)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Loop through the results and display them
    for frame_idx, result in enumerate(results):
        frame = result.plot()  # Get the frame with predictions drawn
        cv2.imshow(window_name, frame)

        if result.keypoints is not None:
            base_tensor = result.keypoints.data
            keypoints = Keypoints(base_tensor, orig_shape=frame.shape)
            normalized_keypoints = np.squeeze(keypoints.xyn.cpu().numpy())
            if len(normalized_keypoints.shape) == 3:
                normalized_keypoints = normalized_keypoints[0]
            elif len(normalized_keypoints.shape) == 2:
                x_norm, y_norm = normalized_keypoints[:, 0], normalized_keypoints[:, 1]
                keypoints_data = {'keypoints': ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'] ,
                                        'x': None * 17,
                                        'y': None * 17}
                for i, keypoint in enumerate(keypoints_data['keypoints']):
                    if i < len(x_norm) and i < len(y_norm):
                        keypoints_data['x'][i] = x_norm[i]
                        keypoints_data['y'][i] = y_norm[i]
                df = pd.DataFrame(keypoints_data)
               
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_source = input("Enter the path to the video file: ")
    results = run_model('yolomodels/best.pt', 'Drowning Detection', input_source)