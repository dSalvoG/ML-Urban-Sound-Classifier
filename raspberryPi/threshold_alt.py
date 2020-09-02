from functions import thedownsample
from functions import highpass
from functions import spl_simply
import librosa
import numpy as np
import glob
import os
import datetime

def thres(parent_dir,sub_dirs,file_ext="*.wav"):
    fn = '/home/pi/Desktop/SSEnCE_merged/audio/input/a-a-audio-0-08_05_2020-3.wav'
    k = np.float64(1.0)
    fs = 16000
    p,s = librosa.load(fn,sr=fs, dtype=k.dtype)
    label = fn.split('-')[3]
    SPL_max_1 = -1
    SPL_max_2 = -1
    i=1
    p,s = librosa.load(fn,sr=fs, dtype=k.dtype)
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(os.path.abspath(parent_dir),sub_dir,file_ext)):
            while True:
                p_hp = highpass(p=p)
                p_spl,p_spl_dB = spl_simply(p, fs=s, windowSize=0.25)
                p_hp_spl,p_hp_spl_dB = spl_simply(p_hp, fs=s, windowSize=0.25)
                SPL_max_1 = np.amax(p_spl)
                SPL_max_2 = np.amax(p_hp_spl)
                print('Number ',i)
                i+=1

    if SPL_max_1 > 0:
        return True
    elif SPL_max_2 > 0:
        return True
    else:
        return False