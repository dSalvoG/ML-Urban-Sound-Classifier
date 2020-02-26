# # LOAD MODELS RASPBERRY PI
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np

print("Tensorflow version: ", tf.__version__)


# LOAD FEATURES AND LABELS
features_test = np.load('features_test1.npy')
labels_test = np.load('labels_test1.npy')
# print(features_test)

# LOAD PRE-TRAINED MODEL
model = tf.keras.models.load_model('models/no1_model.h5')

# MODEL EVALUATION
test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)

