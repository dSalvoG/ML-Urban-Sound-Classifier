# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import pandas as pd
import librosa

import numpy as np

from datetime import datetime 
while(1):
    def get_stream(audio, window_size): # fix the audio data size up to 3s (512*128frames=65536)
        
        if audio.shape[0] < window_size:
            padding = np.zeros(window_size-audio.shape[0])
            stream = np.concatenate((audio,padding), axis=0)

        elif audio.shape[0] >= window_size:
            stream = np.resize(audio,window_size)        

        return stream

    def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 128, frames = 128):
        start = datetime.now()
        window_stream = 512 * (frames - 1) 
        log_specgrams = []
        labels = []
        for l, sub_dir in enumerate(sub_dirs):
            for fn in glob.glob(os.path.join(os.path.abspath(parent_dir),sub_dir,file_ext)):
                sound_clip,s = librosa.load(fn)
                label = fn.split('-')[3]
                stream = get_stream(sound_clip,window_stream)

                melspec = librosa.feature.melspectrogram(stream, sr=s, win_length=512, n_mels = bands)
                logspec = librosa.power_to_db(melspec, ref=np.max)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
                
        log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
        features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
        for i in range(len(features)):
            features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        duration = datetime.now() - start
        print("Feature and label extraction saved in time: ", duration)
        return np.array(features), np.array(labels,dtype = np.int), duration

    parent_dir = 'audio'
    sub_dirs= ['input']
    features,labels,duration = extract_features(parent_dir,sub_dirs)

    # Saving Features and Labels arrays
    # np.save('feature_monitoring', features)
    # np.save('label_monitoring', labels)
    

