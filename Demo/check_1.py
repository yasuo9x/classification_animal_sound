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
audio_fpath = ('../music/fileCut/Mèo/')  
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))
x, sr = librosa.load(audio_fpath+audio_clips[3],44100)
# Hiển thị tín hiểu của file âm thanh
plt.figure(figsize=(14,5))
plt.grid()
librosa.display.waveplot(x,sr=sr)
plt.show()