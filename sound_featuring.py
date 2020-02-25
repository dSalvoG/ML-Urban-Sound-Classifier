import numpy as np

from datetime import datetime 
from functions import extract_features

parent_dir = '../../audio'
sub_dirs= ['fold2']
start = datetime.now()
features,labels = extract_features(parent_dir,sub_dirs)

# Saving Features and Labels arrays
np.save('features_test2', features)
np.save('labels_test2', labels)

duration = datetime.now() - start
print("Feature and label extraction saved in time: ", duration)

