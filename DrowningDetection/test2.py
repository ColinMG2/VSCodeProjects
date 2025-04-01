from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Keypoints
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

video_path = input("Enter the path to the video file: ")
model = YOLO ('yolomodels/yolo11s-pose.pt')
results = model.predict(source=video_path, stream=True)

cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose Detection', 800, 600)

def plot_keypoints(df, keypoints_to_plot):
    num_keypoints = len(keypoints_to_plot)
    fig, axes = plt.subplots(num_keypoints, 1, figsize=(8, 6 * num_keypoints), sharex=True)
    if num_keypoints == 1:
        axes = [axes]
    scatter_plots = {}
    for ax, keypoint in zip(axes, keypoints_to_plot):
        ax.set_title(f'{keypoint.capitalize()} Position')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid()
        scatter_plots[keypoint] = ax.plot([], [], 'o', label=keypoint)[0]
        ax.legend()

    plt.ion()
    return fig, axes, scatter_plots

keypoints_to_plot = ['right knee', 'right wrist', 'left knee', 'left wrist']
fig, axes, scatter_plots = plot_keypoints(pd.DataFrame(), keypoints_to_plot)

keypoint_data = {keypoint: {'x': [], 'y': []} for keypoint in keypoints_to_plot}

for result in results:
    frame = result.plot()
    cv2.imshow('Pose Detection', frame)

    if result.keypoints is not None:
        base_tensor = result.keypoints.data
        keypoints = Keypoints(base_tensor, orig_shape=frame.shape)
        normalized_keypoints = np.squeeze(keypoints.xyn.cpu().numpy())
        if len(normalized_keypoints.shape) == 3:
            normalized_keypoints = normalized_keypoints[0]
        elif len(normalized_keypoints.shape) == 2:
            x_norm, y_norm = normalized_keypoints[:, 0], normalized_keypoints[:, 1]
            keypoints_data = {'keypoints': ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle'] ,
                                      'x': x_norm,
                                      'y': y_norm}
            df = pd.DataFrame(keypoints_data)
            for keypoint in keypoints_to_plot:
                keypoint_x = df.loc[df['keypoints'] == keypoint, 'x'].values[0]   
                keypoint_y = df.loc[df['keypoints'] == keypoint, 'y'].values[0]
                keypoint_data[keypoint]['x'].append(keypoint_x)
                keypoint_data[keypoint]['y'].append(keypoint_y)

                scatter_plots[keypoint].set_data(keypoint_data[keypoint]['x'], keypoint_data[keypoint]['y'])

        for ax in axes:
            ax.relim()
            ax.autoscale_view()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

plt.ioff()
plt.show()
cv2.destroyAllWindows()