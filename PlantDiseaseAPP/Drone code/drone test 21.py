from djitellopy import Tello
import cv2
import time

# Initialize Tello Drone
tello = Tello()

def start_drone_stream():
    """Start the drone stream and return the frame reader."""
    tello.connect()
    tello.streamon()
    time.sleep(2)
    frame_read = tello.get_frame_read()
    print(f"Battery: {tello.get_battery()}%")

    tello.set_video_fps("30")
    tello.set_video_bitrate(Tello.BITRATE_5MBPS)
    tello.set_video_resolution("720")

    return frame_read

def get_video_frame(frame_read):
    """Get the latest video frame."""
    frame = frame_read.frame
    if frame is not None:
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            return jpeg.tobytes()  # Return frame as JPEG bytes
    return None

def move_drone(key):
    """Control drone movement based on key pressed."""
    distance = 20
    if key == ord('w'):
        tello.move_forward(distance)
        print(f"Forward {distance} cm")
    elif key == ord('s'):
        tello.move_back(distance)
        print(f"Backward {distance} cm")
    elif key == ord('a'):
        tello.move_left(distance)
        print(f"Left {distance} cm")
    elif key == ord('d'):
        tello.move_right(distance)
        print(f"Right {distance} cm")
    elif key == ord(' '):
        tello.move_up(distance)
        print(f"Up {distance} cm")
    elif key == ord('x'):
        tello.move_down(distance)
        print(f"Down {distance} cm")
    elif key == 13:  # Enter key
        tello.takeoff()
        print("Takeoff")
    elif key == ord('o'):
        tello.rotate_clockwise(10)
        print("Rotate clockwise")
    elif key == ord('p'):
        tello.rotate_counter_clockwise(10)
        print("Rotate counter-clockwise")
    elif key == ord('q'):
        tello.land()
        print("Landing")

def stop_drone():
    """Stop the drone stream and land it."""
    tello.streamoff()
    tello.land()
    cv2.destroyAllWindows()
