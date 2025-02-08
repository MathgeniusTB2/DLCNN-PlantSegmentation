from djitellopy import Tello
import cv2
import time

# Initialize Tello drone object
tello = Tello()

# Connect to drone and start the video stream
tello.connect()
tello.streamon()
time.sleep(2)

# Get the frame reader
frame_read = tello.get_frame_read()

def get_video_frame():
    """Return a single frame to stream."""
    # Capture the frame from Tello camera feed
    frame = frame_read.frame
    if frame is None:
        return None
    return frame

def stop_stream():
    """Stop the video stream."""
    tello.streamoff()
    tello.land()