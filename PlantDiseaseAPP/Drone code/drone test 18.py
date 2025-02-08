
from djitellopy import Tello
from django.http import StreamingHttpResponse
import cv2
import time

# Initialize the Tello drone
tello = Tello()

# Connect to the Tello drone and start the video stream
try:
    tello.connect()  # Connect to the drone
    tello.streamon()  # Start the video stream
    time.sleep(2)  # Allow the camera to start
    frame_read = tello.get_frame_read()  # Get frame read object from the drone
    print(f"Battery: {tello.get_battery()}%")  # Print battery percentage

    # Define a fixed bounding box (you can change these coordinates based on where you want the bounding box)
    x1, y1, x2, y2 = 100, 100, 500, 500  # Example coordinates for a bounding box (this will be drawn on each frame)
    label = "Healthy"  # Set the initial label to "Healthy" (this can be replaced with the model's label later)
    probability = 0.85  # Example probability, replace with your model's output later

    distance = 20  # Set movement distance for the drone
    tello.set_video_fps("30")  # Set video FPS
    tello.set_video_bitrate(Tello.BITRATE_5MBPS)  # Set video bitrate
    tello.set_video_resolution("720")  # Set video resolution

    while True:
        # Capture a frame from the drone's camera
        frame = frame_read.frame

        if frame is not None:
            # Draw a bounding box (rectangle) on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box with thickness 2

            # Add the label and probability text to the frame (you can change these values later with your model's output)
            cv2.putText(frame, f"{label} ({probability*100:.2f}%)", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display the live frame with the bounding box
            cv2.imshow("Tello Edu Feed", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1) & 0xFF

        # Drone movement controls
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
            print("rotate")
        elif key == ord('p'):
            tello.rotate_counter_clockwise(10)
            print("rotate2")
        elif key == ord('q'):  # Press 'q' to quit the live feed
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    tello.streamoff()  # Stop the video stream
    tello.land()  # Land the drone
    cv2.destroyAllWindows()  # Close the OpenCV window