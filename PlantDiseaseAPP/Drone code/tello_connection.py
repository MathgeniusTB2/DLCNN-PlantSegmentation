from djitellopy import Tello
import time

# Initialize the Tello drone
tello = Tello()

def connect_drone():
    """Connect to the Tello drone and start the video stream."""
    try:
        tello.connect()
        tello.streamon()
        time.sleep(2)
        print(f"Battery: {tello.get_battery()}%")
    except Exception as e:
        print(f"Error connecting to the drone: {e}")

def get_video_frame():
    """Capture and return a frame from the Tello drone's camera."""
    try:
        if not tello.is_connected:
            connect_drone()

        frame_read = tello.get_frame_read()
        frame = frame_read.frame
        return frame
    except Exception as e:
        print(f"Error getting frame from Tello: {e}")
        return None

def stop_drone():
    """Stop the drone stream and land the drone."""
    tello.streamoff()
    tello.land()