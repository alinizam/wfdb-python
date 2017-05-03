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
sig, fields=rdsamp('sampledata/100')
print(sig)
#np.savetxt('c:\\data\\100.txt', sig)
#print(sig.shape)
#print(fields)

from wfdb import plotwfdb

#plotwfdb.plotsigs(sig, fields)


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

plt.figure(1)
plt.subplot(411)
plt.plot(sig[1:2500,1])
plt.ylabel('some numbers')


b, a = signal.butter(50, .25, 'high')
#print(b)

fs = 120.0
lowcut = 1
highcut = 30.0


T = 0.05
nsamples = T * fs
t = np.linspace(0, T, nsamples, endpoint=False)
filteredSignal = butter_bandpass_filter(sig[:,1], lowcut, highcut, fs, order=6)




print("ecgrecord1")
import wfdb

ecgrecord = wfdb.rdsamp('sampledata/100', sampfrom=1200, channels = [1,1])

record = wfdb.rdsamp('sampledata/100', sampto = 1000)
annotation = wfdb.rdann('sampledata/100', 'atr', sampto = 2500)
print(annotation[0])
plt.subplot(412)
plt.plot(filteredSignal[0:2500], label='Derivative of (%g Hz)' )
plt.plot(annotation[0], np.zeros_like(annotation[0]) + 0, 'ro')
#plt.plot(annotation[0])
#wfdb.plotrec(record, annotation = annotation, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')



#from qrs._detect import qrs_detector

#resultlist=qrs_detector(ecgrecord)


plt.subplot(413)
g = abs(np.gradient(filteredSignal,2))
plt.plot(g[0:2500], label='Derivative of (%g Hz)' )


end = 20
ok = 1
m = []
index = end
while (index<10000):
    ortOn = np.average(g[index-10:index ]);
    ortSon= np.average(g[index:index + 10]);

    if ortSon >= 3*ortOn:
        init = index
        m.append(init+ np.argmax(g[init:init+50]))
        index += 50
    index+=1
print(m)
#plt.plot(m[0:2500], np.zeros_like(m[0:2500]) + 0, 'ro')
plt.plot(g[0:2500], label='Derivative of (%g Hz)' )

plt.show()