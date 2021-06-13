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
# lay duong dan va kiem tra os file co trong thu muc
audio_fpath = '../audio/Mèo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
# load thu muc can xet
x, sr = librosa.load(audio_fpath+audio_clips[0],44100)
print(sr) # mac dinh sr = 441000 sample rate
print(x.shape)
# tính thoi gian cua video
print(librosa.get_duration(x, sr))
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)
# plt.show()
# hop_length = 256
# frame_length = 512

# energy = np.array([
#     sum(abs(x[i:i+frame_length]**2))
#     for i in range(0, len(x), hop_length)
# ])
# print(energy.shape)
# rmse = librosa.feature.rms(x, frame_length=frame_length, hop_length=hop_length, center=True)
# rmse.shape
# rmse = rmse[0]
# frames = range(len(energy))
# plt.figure(figsize=(14, 5))
# t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
# librosa.display.waveplot(t)
# plt.show()