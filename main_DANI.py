# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez
# Author: Daniel Sanz Montrull
# # LOAD MODELS RASPBERRY PI

# For audio
# import pyaudio
# import wave
import time
from datetime import datetime 
import os

# For classifier
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from functions import extract_features


# Function for classify
def print_class(parent_dir, sub_dirs):
    # LOAD AND FEATURING
    features,labels = extract_features(parent_dir,sub_dirs)
    # PREDICTION
    start_prediction = datetime.now()

    predicted_vector = model.predict_classes(features)
    predicted_proba_vector = model.predict_proba(features) 
    predicted_proba = predicted_proba_vector[0]
    
    stop_prediction = datetime.now() - start_prediction
    print('Prediction time: ', stop_prediction)

# LOAD PRE-TRAINED MODEL
model = tf.keras.models.load_model('models/no10_model.h5')

class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
               'Drilling', 'Engine Idling', 'Gun Shot', 
               'Jackhammer', 'Siren', 'Street Music']

parent_dir = 'audio'
sub_dirs= ['input']

# while times < num_max:
while(1):
    print_class(parent_dir,sub_dirs)