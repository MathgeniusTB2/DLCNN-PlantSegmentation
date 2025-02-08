import threading
import time
import cv2
import numpy as np
from djitellopy import Tello
from django.http import StreamingHttpResponse
from django.shortcuts import render

# Shared variable to store the current frame
current_frame = None
tello = Tello()

# Function to capture the drone video feed
def capture_video():
    global current_frame
    tello.connect()
    tello.streamon()
    time.sleep(2)  # Give it some time to start streaming
    frame_read = tello.get_frame_read()

    print(f"Battery: {tello.get_battery()}%")

    while True:
        frame = frame_read.frame  # Get the current frame
        if frame is not None:
            current_frame = frame  # Update the shared frame
        time.sleep(0.1)  # Adjust to reduce CPU load

# Function to generate MJPEG stream
def generate_mjpeg():
    global current_frame
    while True:
        if current_frame is not None:
            _, jpeg = cv2.imencode('.jpg', current_frame)
            byte_arr = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + byte_arr + b'\r\n\r\n')

# Django view to stream the video feed
def video_feed(request):
    return StreamingHttpResponse(generate_mjpeg(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

# Control the drone using keyboard input
def control_drone():
    tello.connect()
    tello.takeoff()
    distance = 20  # Distance for movement commands

    while True:
        key = cv2.waitKey(1) & 0xFF  # Wait for keypress

        if key == ord('w'):
            tello.move_forward(distance)
            print(f"Moving forward {distance} cm")
        elif key == ord('s'):
            tello.move_back(distance)
            print(f"Moving backward {distance} cm")
        elif key == ord('a'):
            tello.move_left(distance)
            print(f"Moving left {distance} cm")
        elif key == ord('d'):
            tello.move_right(distance)
            print(f"Moving right {distance} cm")
        elif key == ord(' '):  # Spacebar to move up
            tello.move_up(distance)
            print(f"Moving up {distance} cm")
        elif key == ord('x'):  # Press 'x' to move down
            tello.move_down(distance)
            print(f"Moving down {distance} cm")
        elif key == 13:  # Enter key to take off
            tello.takeoff()
            print("Takeoff")
        elif key == ord('q'):  # Press 'q' to quit
            break

    # Clean up
    tello.streamoff()
    tello.land()
    cv2.destroyAllWindows()

# Start the drone video and control threads
def start_threads():
    # Start video capture thread
    video_thread = threading.Thread(target=capture_video)
    video_thread.daemon = True
    video_thread.start()

    # Start drone control thread
    control_thread = threading.Thread(target=control_drone)
    control_thread.daemon = True
    control_thread.start()