import os
import time
import matplotlib.pyplot as plt
import scipy
import seaborn
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import sklearn

audio_fpath = 'music/fileCut/Mèo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
# sr = librosa.core.get_samplerate(audio_fpath+audio_clips[0]) Ham lay mau sr
x, sr = librosa.load(audio_fpath+audio_clips[0]) ### Co lay sample rate nhung lai theo cach khac can phai sua lai cach load
print(type(x), type(sr))
# print(x.shape, sr)
# plt.rcParams['figure.figsize'] = (11, 5)
# T = 3.0      # duration in seconds
# sr = 22050   # sampling rate in Hertz
# amplitude = np.logspace(-3, 0, int(T*sr), endpoint=False, base=10.0) # time-varying amplitude
# print (amplitude.min(), amplitude.max()) # starts at 110 Hz, ends at 880 Hz
print(librosa.get_duration(x, sr= sr))
X = librosa.fft(x)
X_mag = np.absolute(X)
f = np.linspace(0, sr, len(X_mag))
plt.figure(figsize=(13, 5))
plt.plot(f, X_mag) # magnitude spectrum
plt.xlabel('Frequency (Hz)')