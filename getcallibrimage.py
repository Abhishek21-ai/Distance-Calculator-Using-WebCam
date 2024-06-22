import cv2
import numpy as np
import os

# Define the chessboard dimensions (7 inner corners horizontally, 7 inner corners vertically)
chessboard_size = (7, 7)  
square_size = 1.0  # Define the size of a square in your desired unit (e.g., meters)

# Prepare the object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Create a folder to store the images
if not os.path.exists('calibration_images'):
    os.makedirs('calibration_images')

# Capture images from the webcam
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 923
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
count = 0

while count < 20:  # Capture 20 images
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        cv2.imshow('Calibration', frame)
        
        # Save the captured image
        cv2.imwrite(f'calibration_images/calibration_{count}.jpg', frame)
        count += 1
        cv2.waitKey(500)  # Wait for 500 milliseconds

    cv2.imshow('Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
