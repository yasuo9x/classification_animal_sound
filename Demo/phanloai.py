import os
import time
import numpy
import scipy
import pandas
import sklearn
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from sklearn import neighbors,datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_1 = numpy.load('dactrung/zcr/zcr.npy')
data_2 = numpy.load('dactrung/spectrogram/spectrogram.npy')
data = data_1
data = numpy.concatenate((data,data_2),axis=1)

target_data = numpy.load('dactrung/zcr/target_zcr.npy')

audio_fpath_1 = ('../audio_test/')  
audio_clips_1 = os.listdir(audio_fpath_1)
x, sr = librosa.load(audio_fpath_1+audio_clips_1[3],44100)
print(audio_clips_1[3])
x = x[:127890]

tmp_1 = librosa.feature.zero_crossing_rate(x)[0]
test_1 = numpy.array([tmp_1])

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
test_2 = numpy.reshape(Xdb,(1,256250))

test = test_1;
test = numpy.concatenate((test,test_2),axis=1)

clf = neighbors.KNeighborsClassifier(n_neighbors=6,p=2)
clf.fit(data,target_data)
y_1 = clf.kneighbors(test)
y_pred_1 = clf.predict(test)
print(y_1)
print(y_pred_1)
