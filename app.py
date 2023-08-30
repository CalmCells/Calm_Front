import base64
import os
import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory,Response
from flask_socketio import SocketIO, emit
import random
import dlib
import numpy as np
import time
import matplotlib
import math
matplotlib.use('Agg')  # Set the backend to Agg to enable plotting in threads
import matplotlib.pyplot as plt
import threading
from scipy import signal
from scipy.signal import find_peaks
# from filterpy.kalman import KalmanFilter
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from Stress_DB import ayurstress
from BrightCheck_DB import image_brightness
from Movement_DB import euclidean_distance


app = Flask(__name__, static_folder="./static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)



# Global variable to store previous landmarks
prev_landmarks = None


# Function to find the center point of a set of landmarks
def find_center(landmarks):
    x_sum = sum([p[0] for p in landmarks])
    y_sum = sum([p[1] for p in landmarks])
    center_x = int(x_sum / len(landmarks))
    center_y = int(y_sum / len(landmarks))
    return center_x, center_y

# Function to check for excessive motion in facial landmarks
def check_motion(landmarks, threshold=100000):
    global prev_landmarks
    if prev_landmarks is not None:
        motion = 0
        for i in range(30):
            motion += euclidean_distance(landmarks[i], prev_landmarks[i])
        if motion > threshold:
            return True
    prev_landmarks = landmarks
    return False


def detect_faces(frame):

    True_False = False

    Motion_Flag = False



    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Detect faces using dlib
    faces = detector(gray)


    # Iterate over detected faces
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)
        landmarks_points = []
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            landmarks_points.append((x, y))

        # Check for excessive motion in facial landmarks
        if check_motion(landmarks_points):

            Motion_Flag = True



            # Find the center point of the forehead region
            forehead_landmarks = landmarks_points[17:27]
            center_x, center_y = find_center(forehead_landmarks)

            # Display a message on the forehead
            message = "Keep your face stable"
            cv2.putText(frame, message, (center_x - 100, center_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            

        

        # Draw the boundary of the face
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.polylines(frame, [convexhull], True, (245, 245, 220), 3)

        # Draw the accurate boundaries around eyes, eyebrows, and lips
        left_eye = np.array(landmarks_points[36:42], np.int32)
        right_eye = np.array(landmarks_points[42:48], np.int32)
        left_eyebrow = np.array(landmarks_points[17:22], np.int32)
        right_eyebrow = np.array(landmarks_points[22:27], np.int32)
        lips = np.array(landmarks_points[48:68], np.int32)

        cv2.polylines(frame, [left_eye], True, (255, 255, 255), 1)
        cv2.polylines(frame, [right_eye], True, (255, 255, 255), 1)
        cv2.polylines(frame, [left_eyebrow], True, (255, 255, 255), 1)
        cv2.polylines(frame, [right_eyebrow], True, (255, 255, 255), 1)
        cv2.polylines(frame, [lips], True, (255, 255, 255), 1)

        True_False = True

    return frame, True_False , Motion_Flag


rizz =[]
red_values = []

def capture_rgb_values(frame):

    global red_values


    forehead_region = frame[150:230, 100:540]  # Select forehead region from the frame
    red_values.append(np.mean(forehead_region[:, :, 2]))

def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@socketio.on("connect")
def test_connect():
    #print("Connected")

    emit("my response", {"data": "Connected"})


count =0
count2 =0
bright_holder=[]
rr_avg=0

current_time_client = 0


@socketio.on("current_time")
def get_current_time(current_time):
    global current_time_client

    current_time_client=current_time

    # print("-----***********************current_time_client",current_time)


@socketio.on("image")
def receive_image(image):



    global current_time_client


    global count
    global rizz
    global count2
    global bright_holder
    count+=1

   
 
    
    # Decode the base64-encoded image data
    image = base64_to_image(image)
    frame, face_detected , Motion_Flag = detect_faces(image)
    capture_rgb_values(image)


    Break_Flag = False 

    if current_time_client>24:
        global rizz
        fs = int(len(red_values)/current_time_client)
        rizz = ayurstress(red_values, fs)
       


    if face_detected:
        count2 = 0
    else:
        count2 = count2 + 1

    if count2 > 12:
        Break_Flag = True

        

    if current_time_client>3:


        lighting_data = {'lighting': True}


        if Motion_Flag:
            motion_data = {'motion': False}
        else:
            motion_data = {'motion': True}



        if Break_Flag:
            motion_data = {'motion': False}
        else:
            motion_data = {'motion': True}

        
        combined_data = {**motion_data, **lighting_data}
        # Emit the combined data to all connected clients using the 'data' event
        socketio.emit('data', combined_data)

        #print("--------------**********!!!!!!!!!!!!!This is the data",combined_data)
                

        frame_resized = cv2.resize(frame, (640, 360))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data
        emit("processed_image", processed_img_data)

    else:
        lighting_data = {'lighting': True}
        motion_data = {'motion': True}
        combined_data = {**motion_data, **lighting_data}
        socketio.emit('data', combined_data)


        frame_resized = cv2.resize(frame, (640, 360))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data
        emit("processed_image", processed_img_data)




@app.route("/")
def index():
    return render_template("index.html")



@app.route('/results')
def fshow_results():

    return render_template('results.html', stress_score=rizz[0], stress_level=rizz[1], calm_metrics=rizz[2], calm_features=rizz[3])
# Add the Cache-Control header to all responses
@app.after_request
def add_cache_control(response):
    response.headers["Cache-Control"] = "public, max-age=31536000"
    return response



if __name__ == "__main__":
    socketio.run(app, debug=True)