from functions import thedownsample
from functions import highpass
from functions import spl_simply
import librosa
import numpy as np
import glob
import os
import datetime

def thres(parent_dir,sub_dirs,file_ext="*.wav"):
    start_load = datetime.datetime.now()
    k = np.float64(1.0)
    fs = 16000
    SPL_max_1 = -1
    SPL_max_2 = -1
    print(file_ext)
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(os.path.abspath(parent_dir),sub_dir,file_ext)):
            p,s = librosa.load(fn,sr=fs, dtype=k.dtype)

            stop_load = datetime.datetime.now() - start_load
            print('Load time: ', stop_load)

            start_filters = datetime.datetime.now()

            #p = thedownsample(p=p)

            stop_down = datetime.datetime.now() - start_filters
            #print('         Downsample time: ', stop_down)

            #s = s/3
            p_hp = highpass(p=p)

            stop_hp = datetime.datetime.now() - start_filters - stop_down
            print('         HighPass time: ', stop_hp)

            p_spl,p_spl_dB = spl_simply(p, fs=s, windowSize=0.25)
            p_hp_spl,p_hp_spl_dB = spl_simply(p_hp, fs=s, windowSize=0.25)
            SPL_max_1 = np.amax(p_spl)
            SPL_max_2 = np.amax(p_hp_spl)

            stop_filters = datetime.datetime.now() - start_filters
            print('Filtering time: ', stop_filters)

    if SPL_max_1 > 0:
        return True
    elif SPL_max_2 > 0:
        return True
    else:
        return False