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

audio_fpath = '../music/fileCut/Mèo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
x, sr = librosa.load(audio_fpath+audio_clips[0],44100)
print(sr) # mac dinh sr = 441000 sample rate
print(x.shape)
# Hiển thị tín hiểu của file âm thanh
plt.figure(figsize=(14,5))
plt.grid()
librosa.display.waveplot(x,sr=sr)
# Xác đinh khoảng thời gian trong audio để phóng to lên xử lý
# thoi gian t = x/sr
n0 = 5000 # Bắt đầu
n1 = 44100 # kết thúc
plt.figure(figsize=(14,5))
plt.plot(x[n0:n1]) # neu plt.plot(x) thi co nghia lay ca doan
plt.grid()
# Xac dinh so lan di qua 0 trong 1 khoang thoi gian
zero_crossings = librosa.zero_crossings(x[n0:n1],pad=False)
print(zero_crossings.shape)
print(sum(zero_crossings))

zcrs = librosa.feature.zero_crossing_rate(x+0.0001)
print(zcrs.shape) # tra ve dang mang co bao nhieu chieu moi chieu bao nhieu phan tu
# bieu dien Zero Crossing Rate duoi dang bieu do
plt.figure(figsize=(14, 5))
plt.plot(zcrs[0])
plt.grid()
# ham hien thi
plt.show()


