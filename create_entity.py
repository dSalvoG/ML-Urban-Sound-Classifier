# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez

import json
import datetime
 
currentDT = datetime.datetime.now()
print (str(currentDT))

# passing data classification to json format

data = {
	"type": "Device",
	
	"id": "urn:ngsi-ld:AcousticNode:000",
	
	"source": {
	    "type":"URL",
	    "value":"https://gtac.webs.upv.es/"
	},
	
	"dataProvider": {
	    "type":"URL",
	    "value":"https://gtac.webs.upv.es/"
	},
	
	"category": {
	    "type": "Text",
	    "value": "sensor"
	},
	
	"controlledProperty": {
	    "type": "Text",
	    "value": "noiseClass"
	},
	
	"noiseClass": {
		"type": "Text",
		"value": "Unknown"
	},
	
	"airConditioner":{
		"type": "Number",
		"value": "Unknown"
    },

    "carHorn": {
    	"type": "Number",
    	"value": "Unknown"
    },

    "childrenPlaying":{
    	"type": "Number",
    	"value": "Unknown"
    },

    "dogBark": {
    	"type": "Number",
    	"value": "Unknown"
    },

    "Drilling": {
    	"type": "Number",
    	"value": "Unknown"
    },

    "engineIdling": {
    	"type": "Number",
    	"value": "Unknown"
    },

    "gunShot": {
    	"type": "Number",
    	"value": "Unknown"
    },

    "Jackhammer": {
    	"type": "Number",
    	"value": "Unknown"
    },

    "Siren": {
    	"type": "Number",
    	"value": "Unknown"
    },

    "streetMusic": {
    	"type": "Number",
    	"value": "Unknown"
    },
	
	"function": {
	    "type": "Text",
	    "value": "sensing"
	},
	
	"name": {
	    "type":"Text",
	    "value": "Raspberry David 000"
	},
	
	"address": {
	    "type": "PostalAddress",
	    "value": {
	        "streetAddress": "Camí de Vera, s/n Edificio 8G",
	        "addressRegion": "Valencia",
	        "addressLocality": "Valencia",
	        "postalCode": "46022"
	    },
	    
	    "metadata": {
	        "verified": {
	        	"value": "true",
	            "type": "Boolean"
	        }
	    }
	},
	"location": {
	    "type": "geo:json",
	    "value": {
	         "type": "Point",
	         "coordinates": [39.477313, -0.335811]
	    }
	},

	"Geohash": {
	    "type": "Text",
	    "value": "ezpb86ekp"
	},

	"creDate": {
		"type":"Text",
		"value":str(currentDT)
	},

	"modDate": {
		"type":"Text",
		"value":str(currentDT)
	}
}

## Making a POST request to Orion Broker
import requests

# defining the api-endpoint
API_ENDPOINT = "http://localhost:1026/v2/entities/"

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