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
audio_fpath = ('../music/fileCut/Cho soi/')  
# audio_fpath = '../music/fileCut/Mèo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))

x, sr = librosa.load(audio_fpath+audio_clips[0],44100)
print(sr) # mac dinh sr = 44100Hz sample rate
print(x.shape)
print(librosa.get_duration(x, sr))
# Hiển thị tín hiểu của file âm thanh
plt.figure(figsize=(14,5))
plt.grid()
librosa.display.waveplot(x,sr=sr)
# Xác đinh khoảng thời gian trong audio để phóng to lên xử lý
# thoi gian t = x/sr
n0 = 0 # Bắt đầu n0 = sr * t trong do t1 là thoi diem ma ong muốn bắt đầu xét
n1 = 88200 # kết thúc
plt.figure(figsize=(14,5))
plt.plot(x) # neu plt.plot(x) thi co nghia lay ca doan
plt.grid()
# Xac dinh so lan di qua 0 trong 1 khoang thoi gian
zero_crossings = librosa.zero_crossings(x[n0:n1],pad=False)
print(zero_crossings.shape) # doc python .shape 
print(sum(zero_crossings)) # tinh so lan qua nguong khong trong khoang thoi gian da chon

zcrs = librosa.feature.zero_crossing_rate(x[n0:n1])[0]
print(zcrs.size) # tra ve dang mang co bao nhieu chieu moi chieu bao nhieu phan tu
# bieu dien Zero Crossing Rate duoi dang bieu do
plt.figure(figsize=(14, 5))
plt.plot(zcrs)
plt.grid()

print(zcrs)
# for index in zcrs:
#     print(round(index,3))
# print(round(max(zcrs),3))
# ham hien thi
plt.show()


