import numpy as np
import cv2
import glob
import os

# Define the chessboard dimensions (7 inner corners horizontally, 7 inner corners vertically)
chessboard_size = (7, 7)
square_size = 1.0

# Prepare the object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Load the saved images
images = glob.glob('calibration_images/*.jpg')

# Check if images were loaded
if len(images) == 0:
    print("No images found in the 'calibration_images' directory.")
else:
    print(f"Found {len(images)} images for calibration.")

# Process each image
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        print(f"Corners found in image {fname}")
        cv2.imshow('Found corners', img)
        cv2.waitKey(500)  # Wait for 500 milliseconds to visually inspect the corners
    else:
        print(f"No corners found in image {fname}")
        cv2.imshow('Image without corners', img)
        cv2.waitKey(500)  # Wait for 500 milliseconds to inspect the image

cv2.destroyAllWindows()

# Check if any corners were detected
if len(objpoints) > 0 and len(imgpoints) > 0:
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)
else:
    print("No corners were detected in any image. Calibration cannot be performed.")
