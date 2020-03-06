# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez

import numpy as np

from datetime import datetime 
from functions import extract_features

parent_dir = 'audio'
sub_dirs= ['fold4']
start = datetime.now()
features,labels = extract_features(parent_dir,sub_dirs)

# Saving Features and Labels arrays
np.save('features_test4', features)
np.save('labels_test4', labels)

duration = datetime.now() - start
print("Feature and label extraction saved in time: ", duration)

