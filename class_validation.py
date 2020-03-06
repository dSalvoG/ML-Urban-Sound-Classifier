# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Model

import numpy as np
from functions import extract_features

from datetime import datetime 


# Load pre-trained model
model = tf.keras.models.load_model('models/no10_model.h5')

def print_class(parent_dir, sub_dirs):
    features,labels = extract_features(parent_dir,sub_dirs)
    predicted_vector = model.predict_classes(features)
    print(predicted_vector.size())
    print("Original class:", class_names[labels[0]], '\n') 
    print("The predicted class is:", class_names[predicted_vector[0]], '\n') 
    predicted_proba_vector = model.predict_proba(features) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        print(class_names[i], "\t\t : ", format(predicted_proba[i], '.32f') )

# Class names vector
class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
               'Drilling', 'Engine Idling', 'Gun Shot', 
               'Jackhammer', 'Siren', 'Street Music']

# FOR SUPERVISED CLASIFICATION
features_test = np.load('audio/test_valencia_dataset/features_Valencia.npy')
labels_test = np.load('audio/test_valencia_dataset/labels_Valencia.npy')
# print(features_test.shape)
# print(features_test.dtype)

# test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)

# FOR CLASIFICATION NEW INPUTS
parent_dir = 'audio/UrbanSound_Valencia_dataset'
sub_dirs= ['Horn']
print_class(parent_dir,sub_dirs)

# # Other Example of prediction
classification = model.predict_classes(features_test)

# For big input datashet, show the inputs and classified outputs
for i in range(len(features_test)):
	print("Predicted=%s " % (class_names[prediction[i]]) + ', Real Value=%s ' % (class_names[labels_test[i]]))