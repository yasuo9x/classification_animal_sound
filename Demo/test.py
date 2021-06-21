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

data_1 = numpy.load('dactrung/rmse/rmse.npy')
data_2 = numpy.load('dactrung/zcr/zcr.npy')
data_3 = numpy.load('dactrung/spectrogram/spectrogram.npy')
data = data_1
data = numpy.concatenate((data,data_2),axis=1)
data = numpy.concatenate((data,data_3),axis=1)

target_data_x = numpy.load('dactrung/zcr/target_zcr.npy')
target_data = target_data_x
# target_data_x = numpy.reshape(target_data_x,(1,125))
target_data = numpy.concatenate((target_data,target_data_x),axis=0)
target_data = numpy.concatenate((target_data,target_data_x),axis=0)



audio_fpath_1 = ('../audio_test/')  
audio_clips_1 = os.listdir(audio_fpath_1)
x, sr = librosa.load(audio_fpath_1+audio_clips_1[0],44100)
print(audio_clips_1[0])
x = x[:127890]

tmp_1 = librosa.feature.rms(x)[0]
test_1 = numpy.array([tmp_1])

tmp_2 = librosa.feature.zero_crossing_rate(x)[0]
test_2 = numpy.array([tmp_2])

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
test_3 = numpy.reshape(Xdb,(1,256250))

test = test_1;
test = numpy.concatenate((test,test_2),axis=1)
test = numpy.concatenate((test,test_3),axis=1)
# print(test_1.shape)
# print(test_1)
# print(test_2.shape)
# print(test_2)
# print(test_3.shape)
# print(test_3)
print(data_1)
print(data_1.shape)
print(data_2)
print(data_2.shape)
print(data_3)
print(data_3.shape)
print(data)

# print(data.shape)
# print(target_data)
# print(test.shape)

# clf = neighbors.KNeighborsClassifier(n_neighbors=6,p=2)
# clf.fit(data,target_data)
# y_1 = clf.kneighbors(test)
# y_pred_1 = clf.predict(test)
# print(y_1)
# print(y_pred_1)

