# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez
# Author: Daniel Sanz Montrull

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import pandas as pd
import librosa
import math as m
from scipy import signal

import numpy as np
import time
import datetime

import tensorflow as tf
from tensorflow.keras import models

import json
import requests


# Function for classify
def print_class(p,s,label,model):

    features,labels = extract_features(p,s,label)
    start_pred = datetime.datetime.now()
    class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
               'Drilling', 'Engine Idling', 'Gun Shot', 
               'Jackhammer', 'Siren', 'Street Music']
    predicted_vector = model.predict_classes(features)
    print("The predicted class is:", class_names[predicted_vector[0]], '\n') 
    predicted_proba_vector = model.predict_proba(features) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        print(class_names[i], "\t\t : ", format(predicted_proba[i], '.32f') )
    stop_pred = datetime.datetime.now() - start_pred
    print('Prediction time: ', stop_pred)
    
    currentDT = datetime.datetime.now()
    print (str(currentDT))

    # defining the api-endpoint
    API_ENDPOINT = "http://localhost:1026/v2/entities/urn:ngsi-ld:AcousticNode:000/attrs"

    # passing data classification to json format

    # location_data = {
    #     "location": {
    # 	    "type": "geo:json",
    # 	    "value": {
    # 	         "type": "Point",
    # 	         "coordinates": [39.477861, -0.333295]
    # 	    }
    #    },
    #     "Geohash": {
    #             "type": "geo:json",
    #             "value": "ezpb86tr1"
    #         }
    # }

    data = {	
        "modDate": {
            "type":"Text",
            "value":str(currentDT)
        },

        "noiseClass": {
            "type": "Text",
            "value": class_names[predicted_vector[0]]
        },
        
        "airConditioner":{
            "type": "Number",
            "value": str(predicted_proba[0])
        },

        "carHorn": {
            "type": "Number",
            "value": str(predicted_proba[1])
        },

        "childrenPlaying":{
            "type": "Number",
            "value": str(predicted_proba[2])
        },

        "dogBark": {
            "type": "Number",
            "value": str(predicted_proba[3])
        },

        "Drilling": {
            "type": "Number",
            "value": str(predicted_proba[4])
        },

        "engineIdling": {
            "type": "Number",
            "value": str(predicted_proba[5])
        },

        "gunShot": {
            "type": "Number",
            "value": str(predicted_proba[6])
        },

        "Jackhammer": {
            "type": "Number",
            "value": str(predicted_proba[7])
        },

        "Siren": {
            "type": "Number",
            "value": str(predicted_proba[8])
        },

        "streetMusic": {
            "type": "Number",
            "value": str(predicted_proba[9])
        },
    }


    # headers
    headers_string={
        'Content-Type':'application/json',
        'fiware-service':'openiot',
        'fiware-servicepath':'/'
        }

    # data to be sent to api
    payload = json.dumps(data)


    # request post (using json payload)
    x = requests.post(url= API_ENDPOINT, data= payload, headers= headers_string,)

    #print the response text (the content of the requested file):
    print(x.text)


# Función para hacer el downsample.
def thedownsample(p):
    b = [0.155051025721682,0.310102051443364,0.155051025721682]
    a = [1, -0.620204102886729, 0.240408205773458]
    p_al = signal.lfilter(b,a,p)
    # Diezmado
    p_al = p_al[np.arange(0,p.shape[0],3)]
    return p_al

def highpass(p):
    b = [0.757076375333885,-1.51415275066777,0.757076375333885]
    a = [1,-1.45424358625159,0.574061915083955]
    p_hp = signal.lfilter(b,a,p)
    return p_hp


def spl_simply(p, fs, windowSize, ref = 20e-6):    
    windowSize = min(m.ceil(windowSize*fs), m.ceil(p.shape[0]))
    # Lp_Pa es la longitud del vector p
    L_overlap = m.floor(windowSize/2) # Solapamiento del 50%
    Lavanza = windowSize-L_overlap # Trozo que avanza
    Lp_Pa = p.shape[0]
    # Numero de segmentos resultante donde calculamos SPL (Nseg>=1)
    Nseg = m.floor((Lp_Pa-L_overlap)/Lavanza)
    
    # Cálculo de la presión en rms
    p_rms = np.zeros(Nseg)
    N = np.arange(1,Nseg+1).reshape(Nseg,1)
    T = N*Lavanza/fs

    for n in N:
        Segmento = p[np.arange((n-1)*Lavanza,(n-1)*Lavanza+windowSize)]
        p_rms[n-1] = m.sqrt(np.mean(np.power(Segmento,2.0)))
        # print(p_rms[n-1])
    
    # Calcula la presion en lineal
    spl = p_rms/ref
    return spl

def get_stream(audio, window_size): # fix the audio data size up to 3s (512*128frames=65536)
    
    if audio.shape[0] < window_size:
        padding = np.zeros(window_size-audio.shape[0])
        stream = np.concatenate((audio,padding), axis=0)

    elif audio.shape[0] >= window_size:
        stream = np.resize(audio,window_size)        

    return stream

def extract_features(p, s, label, bands = 128, frames = 128):
    start_ex = datetime.datetime.now()

    fft_size = 512
    overlap_fac = 0.5
    hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
    window_stream = 512 * (frames - 1) 
    log_specgrams = []
    labels = []
    
    stream = get_stream(p,window_stream)
    melspec = librosa.feature.melspectrogram(stream, sr=s, n_fft=fft_size, win_length=512, window='hann', n_mels = bands)
    logspec = librosa.power_to_db(melspec, ref=np.max)
    logspec = logspec.T.flatten()[:, np.newaxis].T
    log_specgrams.append(logspec)
    labels.append(label)
                  
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    
    stop_ex = datetime.datetime.now() - start_ex
    print('Extraction time: ', stop_ex)
    return np.array(features), np.array(labels,dtype = np.int)

def thres(parent_dir,sub_dirs,model,file_ext="*.wav"):
    start_load = datetime.datetime.now()
    
    k = np.float64(1.0)
    fs = 48000
    SPL_max_1 = -1
    SPL_max_2 = -1
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(os.path.abspath(parent_dir),sub_dir,file_ext)):
            p,s = librosa.load(fn,sr=fs, dtype=k.dtype)
            label=fn.split('-')[3]
            
            stop_load = datetime.datetime.now() - start_load
            print('Load time: ', stop_load)

            start_filters = datetime.datetime.now()

            p = thedownsample(p=p)

            stop_down = datetime.datetime.now() - start_filters
            print('Downsample time: ', stop_down)

            s = s/3
            p_hp = highpass(p=p)

            stop_hp = datetime.datetime.now() - start_filters - stop_down
            print('HighPass time: ', stop_hp)

            p_spl = spl_simply(p=p, fs=s, windowSize=0.25)
            SPL_max_1 = np.amax(p_spl)
            
            stop_spl1 = datetime.datetime.now() - start_filters - stop_down - stop_hp
            print('SPL without HP: ', stop_spl1)


            p_hp_spl = spl_simply(p=p_hp, fs=s, windowSize=0.25)
            SPL_max_2 = np.amax(p_hp_spl)

            stop_spl2 = datetime.datetime.now() - start_filters - stop_down - stop_hp - stop_spl1
            print('SPL with HP: ', stop_spl2)

            if SPL_max_1 > 0:
                print_class(p,s,label,model)
                return True
            elif SPL_max_2 > 0:
                print_class(p,s,label,model)
                return True
            else:
                return False