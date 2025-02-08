from djitellopy import Tello
import cv2
import time
import threading
import sys
from django.core.management import execute_from_command_line

# Function to control the drone (keyboard input, movement, etc.)
def drone_control():
    tello = Tello()
    try:
        tello.connect()
        tello.streamon()
        time.sleep(2)  # Give it some time to start streaming
        frame_read = tello.get_frame_read()

        print(f"Battery: {tello.get_battery()}%")

        distance = 20  # Distance for movement commands

        # Main loop to show video feed and control the drone
        while True:
            frame = frame_read.frame

            if frame is not None:
                cv2.imshow("Tello Edu Feed", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

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
            elif key == ord(' '):
                tello.move_up(distance)
                print(f"Up {distance} cm")
            elif key == ord('x'):
                tello.move_down(distance)
                print(f"Down {distance} cm")
            elif key == 13:  # Enter key for takeoff
                tello.takeoff()
                print("Takeoff")
            elif key == ord('o'):
                tello.rotate_clockwise(10)
                print("Rotate clockwise")
            elif key == ord('p'):
                tello.rotate_counter_clockwise(10)
                print("Rotate counterclockwise")
            elif key == ord('q'):  # Press 'q' to exit
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Make sure to stop streaming and land the drone safely
        tello.streamoff()
        tello.land()
        cv2.destroyAllWindows()

# Function to start the drone control in a separate thread
def start_drone_control():
    drone_thread = threading.Thread(target=drone_control)
    drone_thread.daemon = True  # Make sure the thread exits when the main program exits
    drone_thread.start()

# Function to start the Django server
def start_django_server():
    # Start the Django server (the runserver command)
    execute_from_command_line(sys.argv)

# Run the drone control and Django server in parallel
if __name__ == "__main__":
    # Start the drone control in the background
    start_drone_control()

    # Start the Django web server in the foreground
    start_django_server()