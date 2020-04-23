import json
import datetime
 
currentDT = datetime.datetime.now()
print (str(currentDT))

# passing data classification to json format

data = {
	"type": "Device",
	
	"id": "urn:ngsi-ld:AcousticNode:001",
	
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
		"type": "Property",
		"value": "Unknown"
	},
	
	"airConditioner":{
		"type": "Property",
		"value": "Unknown"
    },

    "carHorn": {
    	"type": "Property",
    	"value": "Unknown"
    },

    "childrenPlaying":{
    	"type": "Property",
    	"value": "Unknown"
    },

    "dogBark": {
    	"type": "Property",
    	"value": "Unknown"
    },

    "Drilling": {
    	"type": "Property",
    	"value": "Unknown"
    },

    "engineIdling": {
    	"type": "Property",
    	"value": "Unknown"
    },

    "gunShot": {
    	"type": "Property",
    	"value": "Unknown"
    },

    "Jackhammer": {
    	"type": "Property",
    	"value": "Unknown"
    },

    "Siren": {
    	"type": "Property",
    	"value": "Unknown"
    },

    "streetMusic": {
    	"type": "Property",
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
	         "coordinates": [39.477861, -0.333295]
	    }
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