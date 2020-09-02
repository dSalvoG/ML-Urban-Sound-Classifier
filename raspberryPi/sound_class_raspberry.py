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
    features = extract_features(parent_dir,sub_dirs)
    predicted_vector = model.predict_classes(features)
    # print(predicted_vector.size())
    print('\n')
    # print("Original class:", class_names[labels[0]]) 
    print("The predicted class is:", class_names[predicted_vector[0]], '\n') 
    # Write
    report_file.write("The predicted class is:" + class_names[predicted_vector[0]] + '\n') 

    predicted_proba_vector = model.predict_proba(features) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        # print(i)
        print(class_names[i], "\t\t : ", format(predicted_proba[i], '.32f') )
    report_file.write( "Timestamp:" + str(datetime.datetime.now()) + "\n")
    report_file.write("---------------------------------------------" + "\n")
    report_file.close()
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

# duration = datetime.now() - start
# print('\n')
# print("Classification Duration: ", duration, '\n')
# print(format(predicted[1], '.32f'))
print("The predicted class is (V2):", classP, '\n')

import json
import datetime
 
currentDT = datetime.datetime.now()
print (str(currentDT))

## Making a POST request to Orion Broker
import requests

# defining the api-endpoint
API_ENDPOINT = "http://localhost:1026/v2/entities/urn:ngsi-ld:AcousticNode:001/attrs"

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
		"value": classP
	},
	
	"airConditioner":{
		"type": "Number",
		"value": str(predicted[0])
    },

    "carHorn": {
    	"type": "Number",
    	"value": str(predicted[1])
    },

    "childrenPlaying":{
    	"type": "Number",
    	"value": str(predicted[2])
    },

    "dogBark": {
    	"type": "Number",
    	"value": str(predicted[3])
    },

    "Drilling": {
    	"type": "Number",
    	"value": str(predicted[4])
    },

    "engineIdling": {
    	"type": "Number",
    	"value": str(predicted[5])
    },

    "gunShot": {
    	"type": "Number",
    	"value": str(predicted[6])
    },

    "Jackhammer": {
    	"type": "Number",
    	"value": str(predicted[7])
    },

    "Siren": {
    	"type": "Number",
    	"value": str(predicted[8])
    },

    "streetMusic": {
    	"type": "Number",
    	"value": str(predicted[9])
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
