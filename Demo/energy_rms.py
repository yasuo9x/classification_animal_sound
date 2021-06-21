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
audio_fpath = '../audio/Bo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
# load thu muc can xet
path = "../audio_test/Bo12.wav"
# path = audio_fpath+audio_clips[0]
x, sr = librosa.load(path,44100)
x = x[:127890]

rmse = librosa.feature.rms(x, center=True)[0]
print(rmse.shape)
print(rmse)