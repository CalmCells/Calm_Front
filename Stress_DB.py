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




def calculate_rr_intervals(ppg_data, sampling_rate):
    # Step 1: Perform peak detection on PPG data to identify R-peaks
    peaks, _ = signal.find_peaks(ppg_data, distance=int(sampling_rate * 0.4))

    # Step 2: Calculate the time intervals (RR intervals) between consecutive R-peaks
    rr_intervals = np.diff(peaks) / sampling_rate

    return rr_intervals

def calculate_mean_rr_interval(rr_intervals):
    # Step 3: Calculate the mean RR interval
    mean_rr = np.mean(rr_intervals)

    return mean_rr

def calculate_sdnn(rr_intervals):
    # Step 4: Calculate the standard deviation of RR intervals (SDNN)
    sdnn = np.std(rr_intervals)

    return sdnn

def calculate_rmssd(rr_intervals):
    # Step 5: Calculate the root mean square of successive differences (RMSSD)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))

    return rmssd

def calculate_pnn5(rr_intervals):
    # Step 6: Calculate the proportion of consecutive NN intervals that differ by more than 5 ms (pNN5)
    nn_intervals_diff = np.diff(rr_intervals)
    pnn5 = np.sum(np.abs(nn_intervals_diff) > 0.005) / len(nn_intervals_diff)

    return pnn5

def calculate_lfn_hfn_ratio(ppg_data, sampling_rate):
    # Step 7: Perform spectral analysis to calculate LF and HF powers
    freqs, psd = signal.periodogram(ppg_data, fs=sampling_rate)
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)
    total_power = np.sum(psd)
    lf_power = np.sum(psd[lf_band])
    hf_power = np.sum(psd[hf_band])

    # Step 8: Calculate the normalized LF and HF powers
    lfn = lf_power / total_power
    hfn = (hf_power / total_power)

    return lfn, hfn

def calculate_lf_hf_ratio(lfn, hfn):
    # Step 9: Calculate the LF/HF ratio
    lf_hf_ratio = lfn / hfn

    return lf_hf_ratio



def ayurstress(ppg,fs):

    ppg_data = ppg  # Raw PPG data (list or numpy array)
    sampling_rate = fs  # Sampling rate in Hz





    #removed= 5*fs
    #sampling_rate= len(ppg[int(removed):])/25  # Sampling rate in Hz
    #ppg= ppg[int(removed

    peaks, _ = find_peaks(ppg, distance=int(fs/2))
    bpm = len(peaks)*2

    rr_intervals = calculate_rr_intervals(ppg_data, sampling_rate)
    mean_rr = calculate_mean_rr_interval(rr_intervals)
    sdnn = calculate_sdnn(rr_intervals)
    rmssd = calculate_rmssd(rr_intervals)
    pnn5 = calculate_pnn5(rr_intervals)
    lfn, hfn = calculate_lfn_hfn_ratio(ppg_data, sampling_rate)
    lf_hf_ratio = calculate_lf_hf_ratio(lfn, hfn)



    # Check if the values are above the higher limit, below the lower limit, or within the standard range
    mean_rr_level = "Above Standard" if mean_rr > 1000 else "Below Standard" if mean_rr < 600 else "Within Standard"
    sdnn_level = "Above Standard" if sdnn > 150 else "Below Standard" if sdnn < 50 else "Within Standard"
    rmssd_level = "Above Standard" if rmssd > 50 else "Below Standard" if rmssd < 20 else "Within Standard"
    pnn5_level = "Above Standard" if pnn5 > 0.5 else "Below Standard" if pnn5 < 0.2 else "Within Standard"
    lfn_level = "Above Standard" if lfn > 0.4 else "Below Standard" if lfn < 0.2 else "Within Standard"
    hfn_level = "Above Standard" if hfn > 0.4 else "Below Standard" if hfn < 0.2 else "Within Standard"
    lf_hf_ratio_level = "Above Standard" if lf_hf_ratio > 3 else "Below Standard" if lf_hf_ratio < 1 else "Within Standard"





    stress_score = 0
    calm_metrics= []

    # Check the classification of each feature and update the stress score
    if mean_rr_level == "Above Standard":
        stress_score -= 1
    elif mean_rr_level == "Within Standard":
        stress_score = 0
    elif mean_rr_level == "Below Standard":


        d= "Low mean_rr_level"

        calm_metrics.append(d)

        stress_score += 1

    if sdnn_level == "Above Standard":
        stress_score -= 1
    elif sdnn_level == "Within Standard":
        stress_score += 0
    elif sdnn_level == "Below Standard":


        d= "Low sdnn_level"

        calm_metrics.append(d)

        stress_score += 1

    if rmssd_level == "Above Standard":
        stress_score -= 1
    elif rmssd_level == "Within Standard":
        stress_score = 0
    elif rmssd_level == "Below Standard":


        d= "Low rmssd_level"

        calm_metrics.append(d)

        stress_score += 1

    if pnn5_level == "Above Standard":
        stress_score -= 1
    elif pnn5_level == "Within Standard":
        stress_score = 0
    elif pnn5_level == "Below Standard":



        d= "Low pnn5_level"

        calm_metrics.append(d)

        stress_score += 1

    if lfn_level == "Above Standard":



        d= "High lfn_level"

        calm_metrics.append(d)



        stress_score += 1
    elif lfn_level == "Within Standard":
        stress_score = 0
    elif lfn_level == "Below Standard":
        stress_score -= 1

    if hfn_level == "Above Standard":
        stress_score -= 1
    elif hfn_level == "Within Standard":
        stress_score = 0
    elif hfn_level == "Below Standard":


        d= "Low hfn_level"

        calm_metrics.append(d)



        stress_score += 1

    if lf_hf_ratio_level == "Above Standard":


        d= "High lf_hf_ratio_level"

        calm_metrics.append(d)


        stress_score += 1
    elif lf_hf_ratio_level == "Within Standard":
        stress_score += 0.5
    elif lf_hf_ratio_level == "Below Standard":
        stress_score -= 1

    # Determine stress level based on the stress score
    if stress_score >= 4:
        stress_level = "Very High Stress and Anxiety"
    elif stress_score >= 2:
        stress_level = "High Stress"
    elif stress_score >= 0:
        stress_level = "Moderate Stress"
    elif stress_score >= -2:
        stress_level = "Slightly Disturbed"
    else:
        stress_level = "Calm"




    calm_features =[]
    calm_features =[bpm,mean_rr,sdnn,rmssd,pnn5,lfn,hfn,lf_hf_ratio]


    return stress_score,stress_level,calm_metrics,calm_features



