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

audio_fpath = '../music/fileCut/Cho soi/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
x, sr = librosa.load(audio_fpath+audio_clips[1],44100)
print(audio_clips[4])
print(sr) # mac dinh sr = 441000 sample rate
print(x.shape)
 # ham lay thoi gian cua audio nhe
# Hiển thị tín hiểu của file âm thanh
plt.figure(figsize=(14,5))
plt.grid()
librosa.display.waveplot(x,sr=sr)

X = scipy.fftpack.fft(x)
X_mag = numpy.absolute(X)
f = numpy.linspace(0, sr, len(X_mag))

plt.figure(figsize=(13, 5))
plt.plot(f, X_mag) # magnitude spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude [dB]')
plt.grid()

plt.figure(figsize=(13, 5))
plt.plot(f[:5000], X_mag[:5000])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude [dB]')
plt.grid()

plt.show()