# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez

# # LOAD MODELS RASPBERRY PI
from datetime import datetime 
start = datetime.now()

import tensorflow as tf

from tensorflow.keras import models
import numpy as np

from functions import extract_features

def print_class(parent_dir, sub_dirs):
    features,labels = extract_features(parent_dir,sub_dirs)
    predicted_vector = model.predict_classes(features)
    # print(predicted_vector.size())
    print('\n')
    print("Original class:", class_names[labels[0]]) 
    print("The predicted class is:", class_names[predicted_vector[0]], '\n') 
    predicted_proba_vector = model.predict_proba(features) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        print(class_names[i], "\t\t : ", format(predicted_proba[i], '.32f') )


# LOAD PRE-TRAINED MODEL
model = tf.keras.models.load_model('models/no10_model.h5')

# CLASSIFICATION

class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
               'Drilling', 'Engine Idling', 'Gun Shot', 
               'Jackhammer', 'Siren', 'Street Music']

parent_dir = 'audio'
sub_dirs= ['input']

print_class(parent_dir,sub_dirs)  

duration = datetime.now() - start
print('\n')
print("Classification Duration: ", duration, '\n')
