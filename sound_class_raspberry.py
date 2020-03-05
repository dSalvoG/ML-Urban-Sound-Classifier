# # LOAD MODELS RASPBERRY PI
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np

from datetime import datetime 
from functions import extract_features

parent_dir = 'audio'
sub_dirs= ['fold9']
start = datetime.now()
features,labels = extract_features(parent_dir,sub_dirs)

# Saving Features and Labels arrays
np.save('features_test9', features)
np.save('labels_test9', labels)

duration = datetime.now() - start
print("Feature and label extraction saved in time: ", duration)

# LOAD FEATURES AND LABELS
features = np.load('features_test1.npy')
labels = np.load('labels_test1.npy')

# LOAD PRE-TRAINED MODEL
model = tf.keras.models.load_model('models/no1_model.h5')
# MODEL EVALUATION
test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)