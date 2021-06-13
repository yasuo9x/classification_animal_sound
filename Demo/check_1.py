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

# audio_fpath = ('../music/fileCut/Bò/')
# audio_fpath = ('../music/fileCut/Ca sau/')
# audio_fpath = ('../music/fileCut/Cho soi/')  
# audio_fpath = ('../music/fileCut/Chuột/')
# audio_fpath = ('../music/fileCut/Hổ/')
# audio_fpath = ('../music/fileCut/Hươu/')
# audio_fpath = ('../music/fileCut/Mèo/')
# audio_fpath = ('../music/fileCut/Ngựa/')
# audio_fpath = ('../music/fileCut/Voi/')
audio_fpath = ('../music/fileCut/Vượn/')

audio_clips = os.listdir(audio_fpath)
x = []
for index in range(0,len(audio_clips)) :
    tmp,sr = librosa.load(audio_fpath+audio_clips[index],44100)
    print(audio_clips[index])
    print(tmp.shape)
    print(librosa.get_duration(tmp, sr))
    x.append(tmp) 
zcrs = []
for index in x :
    tmp = librosa.feature.zero_crossing_rate(index)[0]
    print(tmp.shape)
    tmp = tmp[:250]
    zcrs.append(tmp)
    
data = numpy.array(zcrs)
print(data.shape)