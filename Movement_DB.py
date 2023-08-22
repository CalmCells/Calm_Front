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


# Function to calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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