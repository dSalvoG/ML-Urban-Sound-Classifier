from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# models
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np

print(tf.__version__)

# Load pre-trained model
model = tf.keras.models.load_model('models/no10_model.h5')

# Loading Features and Label arrays
features_test = np.load('test_dataset/features_no10_test.npy')
labels_test = np.load('test_dataset/labels_no10_test.npy')
# print(features_test)

# test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)
test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)