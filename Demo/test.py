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
FRAME_SIZE = 2048
HOP_LENGTH = 512
audio_fpath_1 = ('../music/fileCut/Cho soi/')  
audio_clips_1 = os.listdir(audio_fpath_1)

audio_fpath_2 = ('../music/fileCut/Mèo/')
audio_clips_2 = os.listdir(audio_fpath_2)

audio_fpath_3 = ('../music/fileCut/Bò/')
audio_clips_3 = os.listdir(audio_fpath_3)


x_1 = [] # lay chuyen du lieu audio vao mang
x_2 = []
x_3 = []
# print("No. of .wav files in audio folder = ",len(audio_clips_1))
# print("No. of .wav files in audio folder = ",len(audio_clips_2))
# print("No. of .wav files in audio folder = ",len(audio_clips_3))
for index in range(0,len(audio_clips_1)) :
    tmp,sr = librosa.load(audio_fpath_1+audio_clips_1[index],44100)
    x_1.append(tmp)    

for index in range(0,len(audio_clips_2)) :
    tmp,sr = librosa.load(audio_fpath_2+audio_clips_2[index],44100)
    x_2.append(tmp)    

for index in range(0,len(audio_clips_3)) :
    tmp,sr = librosa.load(audio_fpath_3+audio_clips_3[index],44100)
    x_3.append(tmp)

# print(len(x_1))
# print(len(x_2))
# print(sr) # mac dinh sr = 44100Hz sample rate

zcrs_1 = [] # mang chua các phần tử của 1 video có zero crossing
zcrs_2 = []
zcrs_3 = []
for index in x_1 :
    tmp_1 = librosa.feature.zero_crossing_rate(index)[0]
    tmp_1 = tmp_1[:250]
    zcrs_1.append(tmp_1)
# print(zcrs_1[0])
for index in x_2 :
    tmp_2 = librosa.feature.zero_crossing_rate(index)[0]
    tmp_2 = tmp_2[:250]
    zcrs_2.append(tmp_2)
#    zcrs.append(tmp_2)

for index in x_3 :
    tmp_3 = librosa.feature.zero_crossing_rate(index)[0]
    tmp_3 = tmp_3[:250]
    zcrs_3.append(tmp_3)
#    zcrs.append(tmp_3)
# for index in zcrs :
#     print(index.size)
zcrs_x1 = numpy.array(zcrs_1[1:12])
zcrs_x2 = numpy.array(zcrs_2)
zcrs_x3 = numpy.array(zcrs_3)
print(zcrs_x1.shape)
print(zcrs_x2.shape)
print(zcrs_x3.shape)

zcrs = numpy.concatenate((zcrs_x1,zcrs_2))
zcrs = numpy.concatenate((zcrs,zcrs_x3))

test = numpy.array([zcrs_1[0]])

# print(len(data))
# print(test)
# target_data_1 = numpy.array([0] * 12) # gan nhan cho bo data train
# target_data_2 = numpy.array([1] * 13)
# target_data_3 = numpy.array([2] * 12)

target_zcrs = numpy.array([0] * 11 + [1] * 12 + [2] * 11)

clf = neighbors.KNeighborsClassifier(n_neighbors=10,p=2) #
clf.fit(zcrs,target_zcrs)
y = clf.kneighbors(test)
y_pred = clf.predict(test) # du doan lay nhan cua data moi


print(y)
print(y_pred)

