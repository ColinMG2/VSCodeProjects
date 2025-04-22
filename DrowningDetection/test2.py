from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Keypoints
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO ('yolomodels/yolo11s-pose.pt')
results = model.predict(source='videos/shahriar_drowning.mp4', stream=True)

# Create smaller window for displaying yolo results
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Pose Detection', 800, 600)

# Create function to plot x,y positions and FFT of keypoints
def plot_keypoints(df, keypoints_to_plot):
    num_keypoints = len(keypoints_to_plot)
    fig, axes = plt.subplots(num_keypoints, 2, figsize=(8, 4), sharex='col')
    if num_keypoints == 1:
        axes = [axes]
    scatter_plots = {}
    for i, keypoint in enumerate(keypoints_to_plot):
        axes[i][0].set_title(f'{keypoint.capitalize()} change over time')
        axes[i][0].set_xlabel('time (frames)')
        axes[i][0].set_ylabel('X, Y position')
        axes[i][0].grid()
        scatter_plots[f'{keypoint}_x'] = axes[i][0].plot([], [], 'o-', label=f'{keypoint} x')[0]
        scatter_plots[f'{keypoint}_y'] = axes[i][0].plot([], [], 'o-', label=f'{keypoint} y')[0]
        axes[i][0].legend()
        
        axes[i][1].set_title(f'{keypoint.capitalize()} FFT over time')
        axes[i][1].set_xlabel('Frequency (Hz)')
        axes[i][1].set_ylabel('Magnitude (dB)')
        axes[i][1].grid()
        scatter_plots[f'{keypoint}_fft_x'] = axes[i][1].plot([], [], 'o-', label=f'{keypoint} x FFT')[0]
        scatter_plots[f'{keypoint}_fft_y'] = axes[i][1].plot([], [], 'o-', label=f'{keypoint} y FFT')[0]
        axes[i][1].legend()
    
    plt.ion()
    return fig, axes, scatter_plots

# Store keypoints that you want to plot in string list
keypoints_to_plot = ['right ankle', 'left ankle']
fig, axes, scatter_plots = plot_keypoints(pd.DataFrame(), keypoints_to_plot)

# Initialize a dictionary to store keypoint data
keypoint_data = {keypoint: {'x': [], 'y': []} for keypoint in keypoints_to_plot}

# Loop through the results and extract keypoints
for frame_idx, result in enumerate(results):
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
                                      'x': [None] * 17,
                                      'y': [None] * 17}
            for i, keypoint in enumerate(keypoints_data['keypoints']):
                if i < len(x_norm) and i < len(y_norm):
                    keypoints_data['x'][i] = x_norm[i]
                    keypoints_data['y'][i] = y_norm[i]
            df = pd.DataFrame(keypoints_data)

            for keypoint in keypoints_to_plot:
                if keypoint in df['keypoints'].values:
                    keypoint_x = df.loc[df['keypoints'] == keypoint, 'x'].values[0]   
                    keypoint_y = df.loc[df['keypoints'] == keypoint, 'y'].values[0]

                else:
                    keypoint_x = None
                    keypoint_y = None
                    print(f"Keypoint '{keypoint}' not found in the DataFrame.")
                if keypoint_x is not None and keypoint_y is not None:    
                    keypoint_data[keypoint]['x'].append(keypoint_x)
                    keypoint_data[keypoint]['y'].append(keypoint_y)

                    # Create an array of frame numbers
                    frame_array = np.arange(1, len(keypoint_data[keypoint]['x']) + 1)

                    # Calculate FFT for x and y coordinates
                    fft_x = 20 * np.log10(np.abs(np.fft.fft(keypoint_data[keypoint]['x'])))
                    fft_y = 20 * np.log10(np.abs(np.fft.fft(keypoint_data[keypoint]['y'])))
                    freq = np.fft.fftfreq(len(keypoint_data[keypoint]['x']))
                    freq = freq[:len(frame_array) // 2]

                    for i in range(len(keypoints_to_plot)):
                        if len(frame_array) == len(keypoint_data[keypoints_to_plot[i]]['x']):
                            scatter_plots[f'{keypoints_to_plot[i]}_x'].set_data(frame_array, keypoint_data[keypoints_to_plot[i]]['x'])
                            scatter_plots[f'{keypoints_to_plot[i]}_y'].set_data(frame_array, keypoint_data[keypoints_to_plot[i]]['y'])
                            scatter_plots[f'{keypoints_to_plot[i]}_fft_x'].set_data(freq, fft_x[:len(freq)])
                            scatter_plots[f'{keypoints_to_plot[i]}_fft_y'].set_data(freq, fft_y[:len(freq)])

                            axes[i][0].relim()
                            axes[i][0].autoscale_view()
                            axes[i][1].relim()
                            axes[i][1].autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

plt.ioff()
plt.tight_layout()
plt.show()
cv2.destroyAllWindows()