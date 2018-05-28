import scipy.signal as signal


# Creo filtro passa banda con frequenza di taglio 5-15 Hz, è un filtro FIR che ha fase lineare
# ne disegno la risposta in frequenza
# questo filtro introduce un ritardo di 19 campioni
# la funzione riceve in ingresso la freq di campionamento e restituisce i coeff del numeratore
# e denominatore del filtro(b,a)
import scipy.signal as signal
from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt
import numpy as np
from wfdb import rdsamp
from scipy.signal import freqz

import numpy as np
from wfdb import rdsamp
# Örnek verilerin yüklenmesi
sig, fields=rdsamp('sampledata/100')
print(sig)

from wfdb import plotwfdb

# Orjinal işaret çizimi
plt.figure(1)
plt.subplot(411)
plt.plot(sig[1:2500,1])
plt.ylabel('orjinal')


#Filtreleme kısmı
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

b, a = signal.butter(50, .25, 'high')

fs = 120.0
lowcut = 1
highcut = 30.0

T = 0.05
nsamples = T * fs
t = np.linspace(0, T, nsamples, endpoint=False)
filteredSignal = butter_bandpass_filter(sig[:,1], lowcut, highcut, fs, order=4)

print("ecgrecord1")
import wfdb

ecgrecord = wfdb.rdsamp('sampledata/100', sampfrom=1200, channels = [1,1])

record = wfdb.rdsamp('sampledata/100', sampto = 1000)
annotation = wfdb.rdann('sampledata/100', 'atr', sampto = 2500)
print(annotation[0])
plt.subplot(412)
plt.plot(filteredSignal[0:2500], label='Derivative of (%g Hz)' )
plt.plot(annotation[0], np.zeros_like(annotation[0]) + 0, 'ro')
plt.ylabel('Filtre ve MITBIH R')



#Deneme amaçlı birinci türev çizimi
plt.subplot(413)
g = np.gradient(filteredSignal,1)
plt.plot(g[0:2500], label='Derivative of (%g Hz)' )
plt.ylabel('Birinci türev')



#R tepesi bulma 3 örnek arası eğim-türev yaklaşımı

end = 20
ok = 1
m = []
index = end
while (index<10000):
    if  (filteredSignal[index]-filteredSignal[index-3])*(filteredSignal[index+3]-filteredSignal[index])<0:
        m.append(filteredSignal[index])
    else:
        m.append(0)
    index+=1
print(filteredSignal[60:85])
print(m[40:65])

plt.subplot(414)
plt.plot(m[0:2500], label='ECG tepeleri' )
plt.ylabel('Bulunan tepeler')
# Tepeler bir eşikle ayıklanacaktır. O kısım henüz yapılmamıştır.
plt.show()