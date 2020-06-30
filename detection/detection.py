# UNIVERSITAT POLITÉCNICA DE VALÉNCIA
# Author: David Salvo Gutiérrez, Daniel Sanz

import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math as m
import scipy.io

from datetime import datetime 

xo = scipy.io.loadmat('mat/xo.mat', mat_dtype=True, squeeze_me=True)

# audio_path = 'audio/fold1/7383-3-0-1.wav' # dog_bark

# x1 , sr = librosa.load(audio_path, dtype=float64) # Carga las muestras digitales de audio
sr=22050

x = xo['xo']
# print(x)
# print(type(x), type(sr))
# print(sr)
# print(x.size)
# print(x.dtype)

start = datetime.now()
# IIR filter
# Coeficientes [num, den]
a = [1,-0.620204102886729,0.240408205773458]
b = [0.155051025721682,0.310102051443364,0.155051025721682]
# sos1=[0.155051025721682,0.310102051443364,0.155051025721682,1,-0.620204102886729,0.240408205773458]
# x impulso, en su lugar x la que toca
# filtra, y es la señal filtrada
x1 = signal.lfilter(b,a, x)

x_dec =x1[0:88199:2]
fs=int(sr/2)
# print(x1)
# print(x_dec)
# print(x_dec.size)
# print(x_dec)
# print(fs)

## RAMA DERECHA

# Seguimos
# Lp_Pa es la longitud del vector p
# 0.267 segundos en fs
windowSize = 2940
# Solapamiento del 50% 
L_overlap = 1470
# Trozo que avanza 
Lavanza = 1470
# Longitud de xdec: Lxo/3 = 64000
Lxdec = 44100
# Numero de segmentos de 0.250 s que hay en 4 segundos (con solape)
Nseg = 29
# Inicializa LTotal 
LTotal = np.zeros(Nseg)

for n in range(Nseg):
    # print(n)
    aux = x_dec[n*Lavanza:n*Lavanza+windowSize:1]
    LTotal[n] = m.sqrt(np.mean(np.power(aux,2.0)))

# print(LTotal.size)

ref = 20e-6
# Calcula la presion en dB
spl = LTotal/ref; 
# print(spl)

## RAMA IZQUIERDA

# High Filter
# Coeficientes [num, den]
# x impulso, en su lugar x la que toca
# x = signal.unit_impulse(700)
# filtra, y es la señal filtrada
b = [0.757076375333885,-1.51415275066777,0.757076375333885]
a = [1,-1.45424358625159,0.574061915083955]
# x impulso, en su lugar x la que toca
# filtra, y es la señal filtrada
xHP = signal.lfilter(b,a, x_dec)
# xHP = signal.sosfilt(sos2, x_dec)
# plt.plot(xHP, 'k', label='hHP')
# plt.legend(loc='best')
# plt.show()

# Inicializa LHP 
LHP = np.zeros(Nseg)

for n in range(Nseg):
    # print(n)
    aux = xHP[n*Lavanza:n*Lavanza+windowSize:1]
    LHP[n] = m.sqrt(np.mean(np.power(aux,2.0)))

# print(LHP.size)

ref = 20e-6
# Calcula la presion en dB
LHP_r = LHP/ref; 
# print(LHP_r)

stop = datetime.now() - start
print('Load time: ', stop)