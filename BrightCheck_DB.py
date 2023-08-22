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



def pixel_brightness(pixel): # This function taken in one point's pixel values at a time.
    assert 3 == len(pixel)
    r, g, b = pixel
    r1=math.sqrt(0.299 * r * 2)
    g1=math.sqrt(0.587 * r * 2)
    b1=math.sqrt(0.114 * r * 2)
    t1=math.sqrt(0.299 * r * 2 + 0.587 * g * 2 + 0.114 * b ** 2)

    c= [r1,g1,b1,t1]

    return t1



def image_brightness(img):

    # Dimension of img : (80, 440, 3) - To visualize , you can imagine image is a stack of 3 books , lying ek ke upar ke, on a table, and you are looking at them not from above, from putting your eye at the surface level of table , what you will see is - " Three lines having width equal to width of each book"


    nr_of_pixels = len(img) * len(img[0])
    sum1= 0
    for row in img: 

        #Dimension of row : (440 , 3) :  you fr made the book 2d, by stripping away what was not visible when you looked from table's surface pov.

        for pixel in row:

            #Dimension of pixel - 3 : Now you strip away the length of the book and are only concerned with one slice along the length
            # Significance = in a 3,3,3 image. You have taken the r,g,b of 1st point.

            sum1 = sum1+pixel_brightness(pixel)

    img_bright = sum1/nr_of_pixels

    return img_bright

