import os
import time
import matplotlib.pyplot as plt

import librosa
import librosa.display

import IPython.display as ipd
import sklearn

audio_fpath = 'music/fileCut/Mèo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
# sr = librosa.core.get_samplerate(audio_fpath+audio_clips[0]) Ham lay mau sr
x, sr = librosa.load(audio_fpath+audio_clips[0]) ### Co lay sample rate nhung lai theo cach khac can phai sua lai cach load
# print(type(x), type(sr))
# print(x.shape, sr)
y = ipd.Audio(audio_fpath+audio_clips[0])
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.title('Visualizing Audio')

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.title('Spectrogram')
plt.colorbar()

# import numpy as np
# sr = 22050 # sample rate
# T = 5.0    # seconds
# t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
# x = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 Hz
# #Playing the audio
# ipd.Audio(x, rate=sr) # load a NumPy array
# #Saving the audio
# librosa.output.write_wav('tone_220.wav', x, sr)

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
# Computing the time variable for visualization
plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='b')
plt.title('Spectral Centroid')

spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
plt.figure(figsize=(12, 4))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.title('Spectral Rolloff')

plt.show()

