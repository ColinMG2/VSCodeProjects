from moviepy import VideoFileClip
import imageio
import os
import time

def split_video(input_file, output_prefix, clip_duration):
    # Load the video
    video = VideoFileClip(input_file)
    total_duration = video.duration  # Get total duration in seconds
    
    # Calculate number of clips
    num_clips = int(total_duration // clip_duration) + (1 if total_duration % clip_duration > 0 else 0)
    
    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = min((i + 1) * clip_duration, total_duration)  # Ensure it doesn't exceed the total duration
        
        # Extract the subclip
        subclip = video.subclip(start_time, end_time)
        
        # Save the subclip
        output_filename = f"{output_prefix}_part{i+1}.mp4"
        subclip.write_videofile(output_filename, codec="libx264", fps=video.fps)
    
    video.close()
    print("Video splitting complete!")

def extract_frames(input_file, output_dir, fps):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the video
    video = VideoFileClip(input_file)
    
    # Calculate number of frames
    num_frames = int(video.duration * fps)
    
    for i in range(num_frames):
        time = i / fps  # Calculate time in seconds
        
        # Get the frame at the specified time
        frame = video.get_frame(time)
        
        # Save the frame
        output_filename = f"{output_dir}/frame{i+1}.jpg"
        imageio.imwrite(output_filename, frame)
    
    video.close()
    print("Frame extraction complete!")

# Extract frames from video
input_file = 'videos\drowning_001.mp4'
output_dir = 'frames'
fps = 10
extract_frames(input_file, output_dir, fps)