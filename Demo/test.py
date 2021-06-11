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
x_1 = [] # lay chuyen du lieu audio vao mang
x_2 = []
# print("No. of .wav files in audio folder = ",len(audio_clips_1))
# print("No. of .wav files in audio folder = ",len(audio_clips_2))
for index in range(0,len(audio_clips_1)) :
    tmp,sr = librosa.load(audio_fpath_1+audio_clips_1[index],44100)
    x_1.append(tmp)    

for index in range(0,len(audio_clips_2)) :
    tmp,sr = librosa.load(audio_fpath_2+audio_clips_2[index],44100)
    x_2.append(tmp)    

# print(len(x_1))
# print(len(x_2))
# print(sr) # mac dinh sr = 44100Hz sample rate

zcrs_1 = [] # mang chua các phần tử của 1 video có zero crossing
zcrs_2 = []

for index in x_1 :
    tmp_1 = librosa.feature.zero_crossing_rate(index)[0]
    zcrs_1.append(tmp)
# print(zcrs_1[0])
for index in x_2 :
    tmp_2 = librosa.feature.zero_crossing_rate(index)[0]
    zcrs_2.append(tmp)

# for index in zcrs :
#     print(index.size)

data_1 = zcrs_1[1:12] # cho soi
data_2 = zcrs_2 # meo
test = [zcrs_1[0]]
# print(test)
target_data_1 = numpy.array([0] * 11) # gan nhan cho bo data train
target_data_2 = numpy.array([1] * 13)
print(target_data_1)
#target_data_test = [0] Gan nhan cho bo data test thu
# clf = neighbors.KNeighborsClassifier(n_neighbors=1,p=2)
# print(list(target_data_1))
# print(list(target_data_2))

# X_train, X_test, y_train, y_test = train_test_split(data,target,test_size = )
clf_1 = neighbors.KNeighborsClassifier(n_neighbors=11,p=2)
clf_1.fit(data_1,target_data_1)
y_pred_1 = clf_1.predict(test) # du doan lay nhan cua data moi


print(100*accuracy_score([0],y_pred_1))

# print(y_pred_1)
# print(test)

# clf_2 = neighbors.KNeighborsClassifier(n_neighbors=2,p=2)
# clf_2.fit(data_2,target_data_2)
# y_pred_2 = clf_2.predict(test)
# print(100*accuracy_score(target_data_test,y_pred_2))

# print(librosa.get_duration(x, sr))
# # Hiển thị tín hiểu của file âm thanh
# plt.figure(figsize=(12,5))
# plt.grid()
# librosa.display.waveplot(x,sr=sr)

# zcr_x = librosa.feature.zero_crossing_rate(x)[0]
# zcr_x2 = librosa.feature.zero_crossing_rate(x2)[0]
# print(zcr_x.size) 
# print(zcr_x2.size)
# plt.figure(figsize=(14, 5))
# plt.plot(zcr_x)
# plt.plot(zcr_x2)
# plt.grid()

# # print(zcr_x)
# # print(zcrs2)
# # ham hien thi
# plt.show()


