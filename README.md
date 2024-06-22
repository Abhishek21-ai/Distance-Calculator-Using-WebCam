# Distance Calculator Using Webcam

This project utilizes a webcam to calculate the distance between the camera and an object using a YOLOv5 model for object detection and Zhang's algorithm for distance measurement. Below are the detailed instructions on how to replicate this project and a comprehensive description of each file involved.

## Section 1: How to Replicate the Project

1. **Create a Virtual Environment**
   - Using Conda:
     ```bash
     conda create --name distance_calculator_env python=3.8
     ```
   - Using virtualenv:
     ```bash
     virtualenv distance_calculator_env
     ```

2. **Activate the Environment**
   - For Conda:
     ```bash
     conda activate distance_calculator_env
     ```
   - For virtualenv (Linux/MacOS):
     ```bash
     source distance_calculator_env/bin/activate
     ```
   - For virtualenv (Windows):
     ```bash
     .\distance_calculator_env\Scripts\activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   

4. **Perform Camera Callibration**
   First run the getcallibration script and try to captures multiple images of Chessboard from varied angle then run
   the Script perfcallib.py to get camera matrix and replace the camera matrix with the output of the camera matrix in
   app.py
      ```bash
   python getcallibrimage.py
   ```
5. **Run the Flask Server**
   ```bash
   python app.py
   ```

## Section 2: About Script Files

    1. object_detection.py : This script uses a YOLOv5 model to perform object detection on video frames captured by the webcam. It identifies and labels the objects in each frame.
    2. distance_speed.py : This script calculates the distance between the object and the webcam. It uses Zhang's algorithm for distance measurement, leveraging the camera matrix to obtain the focal length and                               using bounding box parameters to calculate the distance.
    3. getcallibrimage.py : This script captures calibration images of a chessboard pattern, which are necessary for camera calibration. The captured images are used to calibrate the camera for accurate distance                              measurement.
    4. perfcallib.py : This script performs calibration on the images stored in the calibration_images directory. It returns the camera matrix and distortion coefficients, which are essential for accurate                                distance measurement.
    5. app.py : This script sets up a Flask web server to serve the video feed from the webcam, overlaying distance calculations on detected objects.
