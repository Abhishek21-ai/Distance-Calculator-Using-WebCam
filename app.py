from flask import Flask, Response, render_template
import cv2
from object_detection import ObjectDetection
from distance_speed import DistanceCalculator
import time
import atexit
import numpy as np

app = Flask(__name__)

FRAME_WIDTH = 1920
FRAME_HEIGHT = 923

camera_matrix = np.array([[928.62488618, 0., 649.66558144],
                          [0., 927.77360897, 366.49067883],
                          [0., 0., 1.]])

dist_coeffs = np.array([[-1.24480530e-01, -3.05654380e+00, -5.67606214e-03, 4.10631979e-03,
                         3.10685054e+01]])

detector = ObjectDetection()
calculator = DistanceCalculator(camera_matrix)
# audio = AudioFeedback()

previous_distance = None
previous_time = None

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def cleanup():
    cv2.destroyAllWindows()

atexit.register(cleanup)

# Create the camera object outside the function
camera = cv2.VideoCapture(0)  # Use the webcam
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

def generate_frames():
    global previous_distance, previous_time, classNames, camera

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        results = detector.detect_objects(frame)
        current_time = time.time()

        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            if conf >= 0.50:  # Ensure confidence threshold is 0.95
                bounding_box = (x1, y1, x2, y2)
                distance = calculator.calculate_distance(bounding_box)
                current_time = time.time()
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)

                # Display class name and confidence
                label = f'Class {classNames[int(cls.item())]} Confidence {conf:.2f} Distance {distance} meters'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

                # Add distance and speed to the bounding box
                if previous_distance is not None and previous_time is not None:
                    speed = abs((previous_distance - distance) / (current_time - previous_time))
                    if distance < 20:  # threshold distance is 20 meters
                        print(distance, speed)
                        # audio.speak(
                        #     f"Vehicle approaching. Distance: {distance:.2f} meters. Speed: {speed:.2f} meters per second.")

                previous_distance = distance
                previous_time = current_time

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global camera
    camera.release()  # Release the webcam capture
    cleanup()         # Close any OpenCV windows
    return 'Video feed stopped'

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
