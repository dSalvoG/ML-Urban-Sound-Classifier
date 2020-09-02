import numpy as np
import librosa

k = np.float64(1.0)
p,s = librosa.load(fn, dtype=k.dtype)

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
        
# Signal filtrada en paso alto es p_hp y la que sólo se ha diezmado es p