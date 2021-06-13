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

# audio_fpath = ('../music/fileCut/Bo/')
# audio_fpath = ('../music/fileCut/CaSau/')
# audio_fpath = ('../music/fileCut/ChoSoi/')  
# audio_fpath = ('../music/fileCut/Chuot/')
# audio_fpath = ('../music/fileCut/Ho/')
# audio_fpath = ('../music/fileCut/Huou/')
# audio_fpath = ('../music/fileCut/Meo/')
# audio_fpath = ('../music/fileCut/Ngựa/')
# audio_fpath = ('../music/fileCut/Voi/')
# audio_fpath = ('../music/fileCut/Vuon/')

audio_fpath_x = '../music/fileCut/'
audio_fpath_y = os.listdir(audio_fpath_x)

for index in range(0,len(audio_fpath_y)) :
    audio_fpath_z = os.listdir(audio_fpath_x + audio_fpath_y[index])
    for index_x in range(0,len(audio_fpath_z)) :
        tmp, sr = librosa.load(audio_fpath_x + audio_fpath_y[index] + '/'+ audio_fpath_z[index_x] , 44100)
        print(tmp.shape)
        print(librosa.get_duration(tmp, sr))




# audio_clips = os.listdir(audio_fpath)
# x = []
# for index in range(0,len(audio_clips)) :
#     tmp,sr = librosa.load(audio_fpath+audio_clips[index],44100)
#     print(audio_clips[index])
#     print(tmp.shape)
#     print(librosa.get_duration(tmp, sr))
#     x.append(tmp) 
# zcrs = []
# for index in x :
#     tmp = librosa.feature.zero_crossing_rate(index)[0]
#     print(tmp.shape)
#     tmp = tmp[:250]
#     zcrs.append(tmp)
    
# data = numpy.array(zcrs)
# print(data.shape)