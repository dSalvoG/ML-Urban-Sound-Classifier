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
        # print(i)
        print(class_names[i], "\t\t : ", format(predicted_proba[i], '.32f') )
    return predicted_proba, class_names[predicted_vector[0]]
    


# LOAD PRE-TRAINED MODEL
model = tf.keras.models.load_model('models/no10_model.h5')

# CLASSIFICATION

class_names = ['Air Conditioner', 'Car Horn', 'Children Playing', 'Dog Bark', 
               'Drilling', 'Engine Idling', 'Gun Shot', 
               'Jackhammer', 'Siren', 'Street Music']

parent_dir = 'audio'
sub_dirs= ['input']

predicted, classP = print_class(parent_dir,sub_dirs)  

duration = datetime.now() - start
print('\n')
print("Classification Duration: ", duration, '\n')
print(format(predicted[1], '.32f'))
print("The predicted class is (V2):", classP, '\n')

import json

# passing data classification to json format

data = {	
	"noiseClass": {
		"type": "Property",
		"value": classP
	},
	
	"airConditioner":{
		"type": "Property",
		"value": str(predicted[0])
    },

    "carHorn": {
    	"type": "Property",
    	"value": str(predicted[1])
    },

    "childrenPlaying":{
    	"type": "Property",
    	"value": str(predicted[2])
    },

    "dogBark": {
    	"type": "Property",
    	"value": str(predicted[3])
    },

    "Drilling": {
    	"type": "Property",
    	"value": str(predicted[4])
    },

    "engineIdling": {
    	"type": "Property",
    	"value": str(predicted[5])
    },

    "gunShot": {
    	"type": "Property",
    	"value": str(predicted[6])
    },

    "Jackhammer": {
    	"type": "Property",
    	"value": str(predicted[7])
    },

    "Siren": {
    	"type": "Property",
    	"value": str(predicted[8])
    },

    "streetMusic": {
    	"type": "Property",
    	"value": str(predicted[9])
    }
}

## Making a POST request to Orion Broker
import requests

# defining the api-endpoint
API_ENDPOINT = "http://localhost:1026/v2/entities/urn:ngsi-ld:AcousticNode:000/attrs"

# headers
headers_string={
    'Content-Type':'application/json'
    }

# data to be sent to api
payload = json.dumps(data)


# request post (using json payload)
x = requests.post(url= API_ENDPOINT, data= payload, headers= headers_string,)

#print the response text (the content of the requested file):
print(x.text)