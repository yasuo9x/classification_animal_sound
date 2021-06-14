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

audio_fpath = '../audio/ChoSoi/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
x, sr = librosa.load(audio_fpath+audio_clips[1],44100)
print(audio_clips[4])
print(sr) # mac dinh sr = 441000 sample rate
print(x.shape)
print(librosa.get_duration(x, sr)) # ham lay thoi gian cua audio nhe
# Hiển thị tín hiểu của file âm thanh
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.grid()

print(type(X))
print(X)
print(X.shape)
print(type(Xdb))
print(Xdb)
print(Xdb.shape)
plt.show()