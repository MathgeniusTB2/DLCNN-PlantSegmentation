import time
import cv2
import threading
from django.http import StreamingHttpResponse
from djitellopy import Tello

# Initialize Tello drone
tello = Tello()

# Function to generate the video feed for streaming response
def gen():
    try:
        tello.connect()  # Connect to drone
        tello.streamon()  # Start video stream
        time.sleep(2)  # Allow time for stream to start
        frame_read = tello.get_frame_read()
        print(f"Battery: {tello.get_battery()}%")

        while True:
            frame = frame_read.frame
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        tello.streamoff()  # Stop streaming
        tello.land()  # Land the drone

# Thread function for controlling the drone (keyboard input)
def drone_control_thread():
    distance = 20  # Distance for movement
    try:
        tello.connect()  # Connect to drone
        tello.streamon()  # Start stream
        time.sleep(2)
        frame_read = tello.get_frame_read()
        print(f"Battery: {tello.get_battery()}%")

        while True:
            frame = frame_read.frame  # Capture frame
            key = cv2.waitKey(1) & 0xFF

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
            elif key == ord(' '):  # Spacebar moves up
                tello.move_up(distance)
                print(f"Up {distance} cm")
            elif key == ord('x'):  # X key moves down
                tello.move_down(distance)
                print(f"Down {distance} cm")
            elif key == 13:  # Enter key for takeoff
                tello.takeoff()
                print("Takeoff")
            elif key == ord('o'):  # O key for clockwise rotation
                tello.rotate_clockwise(10)
                print("Rotate clockwise")
            elif key == ord('p'):  # P key for counter-clockwise rotation
                tello.rotate_counter_clockwise(10)
                print("Rotate counter-clockwise")
            elif key == ord('q'):  # Q key to quit
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        tello.streamoff()
        tello.land()
        cv2.destroyAllWindows()

# View to stream the video feed
def video_feed(request):
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')

# Start the control thread when the app is running
drone_control_thread = threading.Thread(target=drone_control_thread)
drone_control_thread.daemon = True
drone_control_thread.start()