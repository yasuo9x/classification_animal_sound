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
    zcrs = []
    for index_y in x :
        tmp_y = librosa.feature.zero_crossing_rate(index_y)[0]
#        tmp = tmp[:250]
        zcrs.append(tmp_y)
    data_tmp = numpy.array(zcrs)
    if(data.size == 0 ) : 
        data = numpy.array(data_tmp)
    else :
        data = numpy.concatenate((data,data_tmp))
    if(target_data.size == 0 ) :
        target_data = numpy.array([audio_fpath_y[index]] * len(audio_fpath_z))
    else :
        target_data = numpy.append(target_data,[audio_fpath_y[index]] * len(audio_fpath_z))


numpy.save('dactrung/zcr/zcr.npy',data)
numpy.save('dactrung/zcr/target_zcr.npy',target_data)
