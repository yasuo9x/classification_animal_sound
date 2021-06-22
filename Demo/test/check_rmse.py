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


audio_fpath_x = '../audio/'
audio_fpath_y = os.listdir(audio_fpath_x)
data = numpy.array([])
target_data = numpy.array([])
for index in range(0,len(audio_fpath_y)) :
    audio_fpath_z = os.listdir(audio_fpath_x + audio_fpath_y[index])
    x = []
    for index_x in range(0,len(audio_fpath_z)) :
        tmp, sr = librosa.load(audio_fpath_x + audio_fpath_y[index] + '/'+ audio_fpath_z[index_x] , 44100)
        tmp = tmp[:127890]
        x.append(tmp)
    rms = []
    for index_y in x :
        tmp_y = librosa.feature.rms(index_y)[0]
#        tmp = tmp[:250]
        rms.append(tmp_y)
    data_tmp = numpy.array(rms)
    if(data.size == 0 ) : 
        data = numpy.array(data_tmp)
    else :
        data = numpy.concatenate((data,data_tmp))
    if(target_data.size == 0 ) :
        target_data = numpy.array([index] * len(audio_fpath_z))
    else :
        target_data = numpy.append(target_data,[index] * len(audio_fpath_z))


print(data.shape)
print(data)
audio_fpath_1 = ('../audio_test/')  
audio_clips_1 = os.listdir(audio_fpath_1)
x, sr = librosa.load(audio_fpath_1+audio_clips_1[0],44100)
print(audio_clips_1[0])
x = x[:127890]
tmp_1 = librosa.feature.rms(x)[0]
# tmp_1 = tmp_1[:250]
test = numpy.array([tmp_1])

clf = neighbors.KNeighborsClassifier(n_neighbors=5,p=2,weights='distance')
clf.fit(data,target_data)
y = clf.kneighbors(test)
y_pred = clf.predict(test) # du doan lay nhan cua data moi
print(test)
print(y)
print(y_pred)

