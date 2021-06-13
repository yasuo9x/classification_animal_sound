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

audio_fpath = '../audio/Mèo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
x, sr = librosa.load(audio_fpath+audio_clips[4],44100)
print(audio_clips[4])
print(sr) # mac dinh sr = 441000 sample rate
print(x.shape)
print(librosa.get_duration(x, sr)) # ham lay thoi gian cua audio nhe
# Hiển thị tín hiểu của file âm thanh
plt.figure(figsize=(14,5))
plt.grid()
librosa.display.waveplot(x,sr=sr)
# Lay quang pho Spectral Centroid
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print(spectral_centroids.shape)
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r') # normalize for visualization purposes

# Spectral Bandwidth
plt.figure(figsize=(14,5))
plt.grid()
librosa.display.waveplot(x,sr=sr)
spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]
spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]
spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth_2), color='r')
plt.plot(t, normalize(spectral_bandwidth_3), color='g')
plt.plot(t, normalize(spectral_bandwidth_4), color='y')
plt.legend(('p = 2', 'p = 3', 'p = 4'))

# Spectral Contrast
plt.figure(figsize=(14,5))
plt.grid()
spectral_contrast = librosa.feature.spectral_contrast(x, sr=sr)
print(spectral_contrast.shape)
plt.imshow(normalize(spectral_contrast, axis=1), aspect='auto', origin='lower', cmap='coolwarm')

# Spectral Rolloff
plt.figure(figsize=(14,5))
plt.grid()
librosa.display.waveplot(x,sr=sr)
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')

plt.show()