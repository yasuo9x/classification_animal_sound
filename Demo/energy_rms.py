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
audio_fpath = '../music/fileCut/Mèo/'
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
# load thu muc can xet
x, sr = librosa.load(audio_fpath+audio_clips[0])
print(sr)
print(x.shape)
print(librosa.get_duration(x, sr))