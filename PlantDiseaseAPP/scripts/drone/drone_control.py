from djitellopy import Tello
import cv2
import time

# Initialize the Tello drone
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
            cv2.imshow("Tello Edu Feed", frame)

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