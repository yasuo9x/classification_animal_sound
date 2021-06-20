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
audio_fpath = '../audio/Bo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
data = numpy.array([])
for index in range(0,len(audio_clips)-1) :
    tmp, sr = librosa.load(audio_fpath+audio_clips[index],44100)
    print(audio_clips[index])
    # print(tmp.shape)
    # print(librosa.get_duration(tmp, sr))
    tmp = tmp[:127890]
    X = librosa.stft(tmp)
    Xdb = librosa.amplitude_to_db(abs(X))
    # plt.figure(figsize=(14, 5))
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.grid()
    test = numpy.reshape(Xdb,(1,256250))
    if(data.size == 0 ) : 
        data = numpy.array(test)
    else :
        data = numpy.concatenate((data,test))
    print(test)
    print(test.shape)
# plt.show()
audio_fpath = '../audio/Bo/'
audio_clips = os.listdir(audio_fpath)
tmp_1, sr = librosa.load(audio_fpath+audio_clips[10],44100)
print(audio_clips[10])
tmp_1 = tmp_1[:127890]
X_1 = librosa.stft(tmp_1)
Xdb_1 = librosa.amplitude_to_db(abs(X_1))
test_1 = numpy.reshape(Xdb_1,(1,256250))

target_data = numpy.array([0] * 10)

clf = neighbors.KNeighborsClassifier(n_neighbors=9,p=2,weights='distance')
clf.fit(data,target_data)
y = clf.kneighbors(test_1)
y_pred = clf.predict(test_1) # du doan lay nhan cua data moi

print(y)
print(y_pred)