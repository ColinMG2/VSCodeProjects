from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from ultralytics.engine.results import Keypoints

model = YOLO('yolomodels/yolo11m-pose.pt')
results = model.track(source='frame1.jpg', stream=True, show=True)

image = cv2.imread('frame1.jpg')
'''
for result in results:
    if result.keypoints is not None:
        base_tensor = result.keypoints.data
        numpy_array = base_tensor.cpu().numpy()
        print(f'Numpy array structure: \n{numpy_array.shape}')
        numpy_array = np.squeeze(numpy_array)
        print(f'Numpy array structure after squeeze: \n{numpy_array.shape}')
        xy_coords = numpy_array[:, :2]
        print(f'xy_coords structure: \n{xy_coords.shape}')
        x = xy_coords[:, 0]
        y = xy_coords[:, 1]
        keypoints_data = {'keypoints': ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'] ,
                                        'x': x,
                                        'y': y}
        df = pd.DataFrame(keypoints_data)
        print(df)
'''
for result in results:
    if result.keypoints is not None:
        base_tensor = result.keypoints.data
        keypoints = Keypoints(base_tensor, orig_shape=image.shape)
        normalized_keypoints = np.squeeze(keypoints.xyn.cpu().numpy())
        x_norm, y_norm = normalized_keypoints[:, 0], normalized_keypoints[:, 1]
        keypoints_data = {'keypoints': ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'] ,
                                        'x': x_norm,
                                        'y': y_norm}
        df = pd.DataFrame(keypoints_data)
        print(df)
        print(df[df['keypoints'] == 'right knee'])

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cv2.destroyAllWindows()