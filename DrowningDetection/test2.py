from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Keypoints
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = YOLO ('yolomodels/yolo11n-pose.pt')
results = model.predict(source='videos/shahriar_drowning.mp4', show=True)

cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose Detection', 800, 600)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title('Right Knee Position')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
scatter_plot = ax.plot([], [], 'ro', label='right knee')  # Initialize an empty scatter plot
ax.legend()

rk_x = []
rk_y = []

for result in results:
    frame = result.plot()
    cv2.imshow('Pose Detection', frame)

    if result.keypoints is not None:
        base_tensor = result.keypoints.data
        keypoints = Keypoints(base_tensor, orig_shape=frame.shape)
        normalized_keypoints = np.squeeze(keypoints.xyn.cpu().numpy())
        x_norm, y_norm = normalized_keypoints[:, 0], normalized_keypoints[:, 1]
        keypoints_data = {'keypoints': ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'] ,
                                      'x': x_norm,
                                      'y': y_norm}
        df = pd.DataFrame(keypoints_data)
        right_knee_x = df.loc[df['keypoints'] == 'right knee', 'x'].values[0]   
        right_knee_y = df.loc[df['keypoints'] == 'right knee', 'y'].values[0]
        rk_x.append(right_knee_x)
        rk_y.append(right_knee_y)

        scatter_plot.set_data(rk_x, rk_y)  # Update the scatter plot with new data
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
plt.ioff()
plt.show()
cv2.destroyAllWindows()
