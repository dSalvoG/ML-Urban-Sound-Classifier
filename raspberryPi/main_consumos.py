# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez
# Author: Daniel Sanz Montrull
# # LOAD MODELS RASPBERRY PI

# For audio
import pyaudio
import wave
import time
import datetime
import os

# For classifier
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from functions_alt import extract_features
from threshold_alt import thres

# Sample rate: 48 kHz. Resolution: 16 bits. Channel: 1
chunk = 960
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

# num_max: number of wav files to record.
SHORT_RECORD = 1
LONG_RECORD = 4
record_seconds = LONG_RECORD
# 50 min of recordings with 4 seconds
num_max = 30
times = 0
condition = False

# Function for classify
def print_class(parent_dir, sub_dirs):

    features,labels = extract_features(parent_dir,sub_dirs)
    
    
#     start_pred = datetime.datetime.now()
# 
#     predicted_vector = model.predict_classes(features)
#     predicted_proba_vector = model.predict_proba(features) 
#     predicted_proba = predicted_proba_vector[0]
# 
#     stop_pred = datetime.datetime.now() - start_pred
#     print('Prediction time: ', stop_pred)

# LOAD PRE-TRAINED MODEL
model = tf.keras.models.load_model('models/no10_model.h5')

class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
               'Drilling', 'Engine Idling', 'Gun Shot', 
               'Jackhammer', 'Siren', 'Street Music']

parent_dir = 'audio'
sub_dirs= ['input']

while times < num_max:

#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT,
#                     channels=CHANNELS,
#                     rate=RATE,
#                     input=True,
#                     output=True,
#                     frames_per_buffer=chunk)
#     
#     frames = []
#     print(" times = %i" % times)
# 
#     for i in range(0, int(RATE / chunk * record_seconds)):
#         data = stream.read(chunk)
#         frames.append(data)
#     
#     todaydate = datetime.date.today()
#     today = todaydate.strftime("%d_%m_%Y")
#     file_name_with_extension = "a-a-audio-0-" + today + "-" + str(times) + ".wav"
# 
#     stream.stop_stream()
#     stream.close()
#     p.terminate()
# 
#     wf = wave.open("audio/input/" + file_name_with_extension, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(2)
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()
#     print_class(parent_dir,sub_dirs)
    # If the signal raise
#     if condition:
    print_class(parent_dir,sub_dirs)
#         record_seconds = SHORT_RECORD
#         condition = False
#     else:
#     thres(parent_dir=parent_dir,sub_dirs=sub_dirs)
#         if condition:
#             record_seconds = LONG_RECORD

#     remove_path = "/home/pi/Desktop/SSEnCE_merged/audio/input/"+file_name_with_extension
#     os.remove(remove_path)
    times += 1