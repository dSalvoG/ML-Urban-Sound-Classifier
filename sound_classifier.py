#!/usr/bin/env python
# coding: utf-8

# # Load Models on Raspberry Pi

# In[1]:
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
# print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# models
from tensorflow.keras import datasets, layers, models
import numpy as np

print(tf.__version__)


# In[2]:
# Load pre-trained model
model = tf.keras.models.load_model('models/no10_model.h5')


# # Use the trained model to make predictions
# We've trained a model and "proven" that it's good—but not perfect—at classifying Iris species. Now let's use the trained model to make some predictions on unlabeled examples; that is, on examples that contain features but not a label.
# 
# In real-life, the unlabeled examples could come from lots of different sources including apps, CSV files, and data feeds. For now, we're going to manually provide three unlabeled examples to predict their labels. Recall, the label numbers are mapped to a named representation as:

# In[5]:
# Loading Features and Label arrays
features_test = np.load('features_test10.npy')
labels_test = np.load('labels_test10.npy')
# print(features_test)


# In[6]:
# test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)
test_loss, test_acc = model.evaluate(features_test, labels_test, verbose=2)

