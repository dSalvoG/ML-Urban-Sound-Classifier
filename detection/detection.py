# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import pandas as pd
import librosa

import numpy as np

from datetime import datetime 

audio,s = librosa.load('audio/audio_filter/prueba.waw')

print(audio.size)