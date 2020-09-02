#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:15:39 2020

@author: gemapinero
"""

# Mirar
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html#scipy.signal.sosfilt

# import matplotlib.pyplot as plt
from scipy import signal

# Coeficientes [num, den]
sos2 = [0.757076375333885,-1.51415275066777,0.757076375333885,1,-1.45424358625159,0.574061915083955]
# x impulso, en su lugar x la que toca
x = signal.unit_impulse(700)
# filtra, y es la señal filtrada
y_sos = signal.sosfilt(sos2, x)
# plt.plot(y_sos, 'k', label='hHP')
# plt.legend(loc='best')
# plt.show()
