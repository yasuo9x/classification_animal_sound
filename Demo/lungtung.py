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


print(target_data)
print(target_data.shape)
print(target_data_x)
print(target_data_x.shape)