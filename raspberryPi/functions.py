# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez
# Author: Daniel Sanz Montrull

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import pandas as pd
import librosa
import math as m

import numpy as np

import datetime

# Función para hacer el downsample.
def thedownsample(p):
    p_al = np.zeros(p.shape)
    G = 0.155051025721682
    b0 = 1
    b1 = 2
    b2 = 1
    a0 = 1
    a1 = -0.620204102886729
    a2 = 0.240408205773458
    for n in np.arange(p.shape[0]):
        if n == 0:
            p_al[n] = G*b0*p[n]
        elif n == 1:
            p_al[n] = G*(b0*p[n]+b1*p[n-1])-a1*p_al[n-1]
        else:
            p_al[n] = G*(b0*p[n]+b1*p[n-1]+b2*p[n-2])-a1*p_al[n-1]-a2*p_al[n-2]
    # Diezmado
    diezmado = np.arange(0,p.shape[0],3)
    p = p_al[diezmado]
    return p

def highpass(p):
    p_hp = np.zeros(p.shape)
    G = 0.757
    b0 = 1
    b1 = -2
    b2 = 1
    a0 = 1
    a1 = -1.4542
    a2 = 0.5741
    for n in np.arange(p.shape[0]):
        if n == 0:
            p_hp[n] = G*b0*p[n]
        elif n == 1:
            p_hp[n] = G*(b0*p[n]+b1*p[n-1])-a1*p_hp[n-1]
        else:
            p_hp[n] = G*(b0*p[n]+b1*p[n-1]+b2*p[n-2])-a1*p_hp[n-1]-a2*p_hp[n-2]
    return p_hp


def spl_simply(p, fs=-1, ref = 20e-6, windowSize = 0):
    # Salta un error si no se pasa el vector p con la presión sonora.    
    if p is None:
        print("Se debe de pasar un vector p")
        return 0
    if  windowSize <= 0 and fs==-1:
        windowSize = min(1024, p.shape[0])
    elif fs == -1:
        windowSize = min(windowSize, p.shape[0])
    else:
        windowSize = min(m.ceil(windowSize*fs), m.ceil(p.shape[0]))
        
    # Seguimos
    # Lp_Pa es la longitud del vector p
    L_overlap = m.floor(windowSize/2) # Solapamiento del 50%
    Lavanza = windowSize-L_overlap # Trozo que avanza
    Lp_Pa = p.shape[0]
    # Numero de segmentos resultante donde calculamos SPL (Nseg>=1)
    Nseg = m.floor((Lp_Pa-L_overlap)/Lavanza)
    
    # Cálculo de la presión en rms
    p_rms = np.zeros(Nseg)
    N = np.arange(1,Nseg+1).reshape(Nseg,1)
    if fs != -1:
        T = N*Lavanza/fs
    else:
        T = N*Lavanza
    for n in N:
        Segmento = p[np.arange((n-1)*Lavanza,(n-1)*Lavanza+windowSize)]
        p_rms[n-1] = m.sqrt(np.mean(np.power(Segmento,2.0)))
        # print(p_rms[n-1])
    
    # Calcula la presion en lineal
    spl = p_rms/ref
    # Calcula la presion en dB
    spl_dB = 20.0 * np.log10(spl) 
    return spl, spl_dB

def get_stream(audio, window_size): # fix the audio data size up to 3s (512*128frames=65536)
    
    if audio.shape[0] < window_size:
        padding = np.zeros(window_size-audio.shape[0])
        stream = np.concatenate((audio,padding), axis=0)

    elif audio.shape[0] >= window_size:
        stream = np.resize(audio,window_size)        

    return stream

def extract_features(parent_dir,sub_dirs,file_ext="*.wav",bands = 128, frames = 128):

    fft_size = 512
    overlap_fac = 0.5
    hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
    window_stream = 512 * (frames - 1) 
    log_specgrams = []
    labels = []
    k = np.float64(1.0)
    start_ex = 100
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(os.path.abspath(parent_dir),sub_dir,file_ext)):
            # fs = 48000
            fs = 22050
            p,s = librosa.load(fn,sr=fs, dtype=k.dtype)
            label = fn.split('-')[3]   
#             p_spl,p_spl_dB = spl_simply(p, fs=s, windowSize=0.25)
#             SPL_max = np.amax(p_spl)

            stream = get_stream(p,window_stream)
            
            melspec = librosa.feature.melspectrogram(stream, sr=s, n_fft=fft_size, win_length=512, window='hann', n_mels = bands)
            logspec = librosa.power_to_db(melspec, ref=np.max)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)
            labels.append(label)
                  
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return np.array(features), np.array(labels,dtype = np.int)

