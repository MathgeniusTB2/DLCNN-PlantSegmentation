from django.http import StreamingHttpResponse
from djitellopy import Tello
import cv2
import time

# Initialize the Tello drone
tello = Tello()


# Function to generate video frames for streaming
def gen():
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
                cv2.putText(frame, f"{label} ({probability * 100:.2f}%)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Convert the frame to JPEG format for streaming
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        tello.streamoff()  # Stop the video stream
        tello.land()  # Land the drone


def video_feed(request):
    """Stream the video feed from Tello."""
    return StreamingHttpResponse(gen(), content_type='multipart/x-mixed-replace; boundary=frame')