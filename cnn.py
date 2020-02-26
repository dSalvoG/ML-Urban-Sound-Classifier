from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from datetime import datetime 

from sklearn import model_selection
from sklearn.model_selection import train_test_split 

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

# LOAD FEATURES AND LABELS
features = np.load('../../audio/train_dataset/features_no5.npy')
labels = np.load('../../audio/train_dataset/labels_no5.npy')

# SPLIT DATASET: set into training and test
validation_size = 0.20
seed = 42
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=validation_size, random_state=seed)

# CONSTRUCT MODEL 
def create_model():
    start = datetime.now()
    model = models.Sequential()

    model.add(layers.Conv2D(24, (5, 5), activation='relu', input_shape=(128, 128, 2)))
    model.add(layers.MaxPooling2D((4, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(48, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((4, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(48, (5, 5), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    duration = datetime.now() - start
    print("Model compilation completed in time: ", duration)
    # model.summary()
    return model

# LOAD MODEL
# model = tf.keras.models.load_model('models/no10_model.h5')

# CREATE MODEL 
model=create_model()
model.summary()

# TRAINING
model.fit(X_train, Y_train, epochs=25, validation_data=(X_test, Y_test))

# from sklearn.model_selection import KFold
# n_split=10

# X=features
# Y=labels
# for train_index,test_index in KFold(n_split).split(X):
#   x_train,x_test=X[train_index],X[test_index]
#   y_train,y_test=Y[train_index],Y[test_index]
  
#   model=create_model()
#   model.fit(x_train, y_train,epochs=25, validation_data=(x_test, y_test))
  
#   print('Model evaluation ',model.evaluate(x_test,y_test))


# EVALUATION
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)

train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)

print("Training Accuracy: ", train_score[1])
print("Testing Accuracy: ", test_score[1])

# Save Model
# model.save('models/no5_model_25epchs.h5')
