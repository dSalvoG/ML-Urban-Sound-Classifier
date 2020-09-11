# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez
# Author: Daniel Sanz Montrull
# # LOAD MODELS RASPBERRY PI

# For audio
import pyaudio
import wave
import time
import datetime
import os

# For classifier
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
from functions import thres

# Sample rate: 48 kHz. Resolution: 16 bits. Channel: 1
chunk = 960
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

# num_max: number of wav files to record.
record_seconds = 4.064
# 50 min of recordings with 4 seconds
num_max = 1000
times = 0
condition = False
# LOAD PRE-TRAINED MODEL
model = tf.keras.models.load_model('models/no10_model.h5')

parent_dir = 'audio'
sub_dirs= ['input']

while times < num_max:
    start_rec = datetime.datetime.now()

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    
    frames = []
    print(" times = %i" % times)

    for i in range(0, int(RATE / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stop_rec =  datetime.datetime.now() - start_rec
    print('Recording time: ', stop_rec)


    start_save = datetime.datetime.now()
    
    todaydate = datetime.date.today()
    today = todaydate.strftime("%d_%m_%Y")
    file_name_with_extension = "audio-0-" + today + "-" + str(times) + ".wav"

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("audio/input/" + file_name_with_extension, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    stop_save =  datetime.datetime.now() - start_save
    print('Saving time: ', stop_save)

    thres(parent_dir=parent_dir,sub_dirs=sub_dirs,model=model)

    start_clean = datetime.datetime.now()

    remove_path = "/home/pi/ml-exercise/SSEnCE-merged/audio/input/"+file_name_with_extension
    os.remove(remove_path)

    stop_clean = datetime.datetime.now() - start_clean
    print('Clean time: ', stop_clean)

    print('\n')
    times += 1