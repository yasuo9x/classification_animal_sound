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
numpy.savetxt('dactrung/data_1.csv', data, delimiter=',')