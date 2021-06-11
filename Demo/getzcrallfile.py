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
audio_fpath = ('../music/fileCut/Cho soi/')  
audio_clips = os.listdir(audio_fpath)
x = []
print("No. of .wav files in audio folder = ",len(audio_clips))
for index in range(0,len(audio_clips)) :
    tmp,sr = librosa.load(audio_fpath+audio_clips[index],44100)
    x.append(tmp)    
print(sr) # mac dinh sr = 44100Hz sample rate
zcrs_1 = []
for index in x :
    tmp = librosa.feature.zero_crossing_rate(index)[0]
    zcrs_1.append(tmp)
# for index in zcrs :
#     print(index.size)
data = zcrs_1[1:12]
test = [zcrs_1[0]]
target_data = range(len(data))
# clf = neighbors.KNeighborsClassifier(n_neighbors=1,p=2)
print(list(target_data))
#X_train, X_test, y_train, y_test = train_test_split(data,target,test_size = )
clf = neighbors.KNeighborsClassifier(n_neighbors=11,p=2)
clf.fit(data,target_data)
y_pred = clf.predict(test)
print(100*accuracy_score([0],y_pred))
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



