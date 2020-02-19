#!/usr/bin/env python
# coding: utf-8

# # Urban Sound Classification, Part 1
# ## Feature extraction from sound
# 
# ### Introduction
# We all got exposed to different sounds every day. Like, the sound of traffic jam, siren, music and dog bark etc. We understand sound in terms of recognition and association to a known sound, but what happens when you don't know that sound and not recognize his source?
# 
# Well that's the same starting point for a computer classifier. In that sense, how about teaching computer to classify such sounds automatically into different categories.
# 
# In this notebook we will learn techniques to classify urban sound using machine learnig. Classifying sound is pretty different from other source of data. In this notebook we will first see what features can be extracted from sound data and how easy it is to extract such features in Python using open source library called [*Librosa*](http://librosa.github.io/).
# 
# To follow this tutorial, make sure you have installed the following tools:
# * Tensorflow
# * Librosa
# * Numpy
# * Matplotlib
# * glob
# * os
# * keras
# * pandas
# * scikit-learn
# * datetime

# ### Dataset
# 
# In this experience we are focused on delvelop a convolutional neural network model able to classify automatically the different urban sounds. To train and work afford this project, I use the urban sound dataset UbanSound8K published by the SONY project reserchers. It contains 8.732 labelled sound clips (<= 4s) from ten different classes according their Urban Soun Taxonomy publication:
# 
# * Aire Conditioner (classID = 0)
# * Car Horn (classID = 1)
# * Children Playing (classID = 2)
# * Dog Bark (classID = 3)
# * Drilling (classID = 4)
# * Engine Idling (classID = 5)
# * Gun Shot (classID = 6)
# * Jackhammer (classID = 7)
# * Siren (classID = 8)
# * Street Music (classID = 9)
# 
# This dataset is available for free in this link, [*UrbanSound8K*](https://urbansounddataset.weebly.com/urbansound8k.html).
# 
# Whe you download the dataset, you will get a '.tar.gz' compressed file (UNIX compression distribution), from Windows you can use prgrams like 7-zip to uncompress the file.
# 
# That file contains two different directories, one of them you can find information about audio fragments classification from a metadata 'UrbanSound8K.csv' file. The other directory contains the audio segments divided in 10 different blocks not classified by classes. Finally, audio data is distibuted as:
# 
# * slice_file_name: The name of the audio file. The name takes the following format: fsID-classID-occurrenceID-sliceID.wav
# * fsID: the Freesound ID of the recording from which this excerpt (slice) is taken
# * start: The start time of the slice in the original Freesound recording
# * end: The end time of slice in the original Freesound recording
# * salience: A (subjective) salience rating of the sound. 1 = foreground, 2 = background.
# * fold: The fold number (1-10) to which this file has been allocated.
# * classID: A numeric identifier of the sound class
# * class: The class name
# 
# source:
# J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
# 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

# ## Feature Extraction
# Even system that use feature learning need to first transform the signal into a representation that lends itself to successful learning. For audio signals,a popular representation is the mel-spectrogram.
# 
# The mel-spectogram is obtained by taking the short-time Fourier transform and mapping its spectral magnitude onto the perceptually motivated mel-scale using a filterbak in the frequency domain. It is the starting point for computin MFCCs, and a popular representation for many audio analysis algorithms including ones based on unsupervised feature learning.

# In[2]:


import glob
import os
import pandas as pd
import librosa
from datetime import datetime 

import numpy as np


# to extract log-scaled
# mel-spectrograms with 128 components (bands) covering the
# audible frequency range (0–22 050 Hz), using a window size of
# 23 ms (1024 samples at 44.1 kHz) and a hop size of the same
# duration. Since the excerpts in our evaluation dataset (described
# below) are of varying duration (up to 4 s), we fix the size of the
# input TF-patch X to 3 s (128 frames), i.e., X ∈ R128×128. TFpatches
# are extracted randomly (in time) from the full log-melspectrogram
# of each audio excerpt during training as described
# below.

# In[5]:


## Load metadata set into pd DataFrame to take the label value for each sound clip


# In[2]:


def get_stream(audio, window_size): # fix the audio daya size up to 3s (512*128frames=65536)
    
    if audio.shape[0] < window_size:
        padding = np.zeros(window_size-audio.shape[0])
        stream = np.concatenate((audio,padding), axis=0)
#         print('shape menor', stream.shape)
    elif audio.shape[0] >= window_size:
        stream = np.resize(audio,window_size)
#         print('shape mayor', stream.shape)
        
    return stream

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

# to extract log-scaled mel-spectrogram with 128 bands covering audible freq (0-22.050Hz)
# using windows_size of 23 ms (1024 samples at 44.1 kHz) and a hop size of the same duration
# fix the input TF-patch X to 3s (128 frames)
def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 128, frames = 128):
    window_stream = 512 * (frames - 1) # fix all audio input up to 3 seconds
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
# Cambiar codigo de forma que se obtenga el valor de la etiqueta
# desde el archivo .csv "metadata" y no desde el nombre de archivo
        for fn in glob.glob(os.path.join(os.path.abspath(parent_dir),sub_dir,file_ext)):
            sound_clip,s = librosa.load(fn)
            label = fn.split('-')[3]
            stream = get_stream(sound_clip,window_stream)

            # revisar el atributo win_length=n_fft/2=1024 de los 2048 de n_fft
            melspec = librosa.feature.melspectrogram(stream, sr=s, win_length=512, n_mels = bands) #revisar n_mels a 128
            logspec = librosa.power_to_db(melspec, ref=np.max)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
            labels.append(label)
            
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    
    return np.array(features), np.array(labels,dtype = np.int)


# In[3]:


parent_dir = 'audio'
sub_dirs= ['fold1']
start = datetime.now()
features,labels = extract_features(parent_dir,sub_dirs)

# Saving Features and Labels arrays
np.save('features_test1', features)
np.save('labels_test1', labels)

duration = datetime.now() - start
print("Feature and label extraction saved in time: ", duration)

