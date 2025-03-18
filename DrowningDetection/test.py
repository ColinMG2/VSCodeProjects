from ultralytics import YOLO

# Load model
model = YOLO('yolo11m-pose.pt')

# Results
results = model.track(source='videos\shahriar_drowning.mp4', show=True)