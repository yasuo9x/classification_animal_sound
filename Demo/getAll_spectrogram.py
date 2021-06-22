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
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# audio_fpath = ('../audio/Bo/')
# audio_fpath = ('../audio/CaSau/')
# audio_fpath = ('../audio/ChoSoi/')
# audio_fpath = ('../audio/Chuot/')
# audio_fpath = ('../audio/Ho/')
# audio_fpath = ('../audio/Huou/')
# audio_fpath = ('../audio/Meo/')
# audio_fpath = ('../audio/Ngựa/')
# audio_fpath = ('../audio/Voi/')
# audio_fpath = ('../audio/Vuon/')
audio_fpath_x = '../audio/'
audio_fpath_y = os.listdir(audio_fpath_x)
data = numpy.array([])
target_data = numpy.array([])
for index in range(0, len(audio_fpath_y)):
    audio_fpath_z = os.listdir(audio_fpath_x + audio_fpath_y[index])
    x = []
    for index_x in range(0, len(audio_fpath_z)):
        tmp, sr = librosa.load( audio_fpath_x + audio_fpath_y[index] + '/' + audio_fpath_z[index_x], 44100)
        tmp = tmp[:127890]
        x.append(tmp)
    spec = []
    for index_y in x:
        tmp_x = librosa.stft(index_y)
        tmp_db = librosa.amplitude_to_db(abs(tmp_x))
        tmp_reshape = numpy.reshape(tmp_db, (1, 256250))
        tmp = tmp[:250]
        spec.append(tmp_reshape)
    data_tmp = numpy.array(spec)
    if(data.size == 0):
        data = numpy.array(data_tmp)
    else:
        data = numpy.concatenate((data, data_tmp))
    if(target_data.size == 0):
        target_data = numpy.array([audio_fpath_y[index]] * len(audio_fpath_z))
    else:
        target_data = numpy.append(target_data, [audio_fpath_y[index]] * len(audio_fpath_z))

data = numpy.reshape(data,(len(target_data),256250))

numpy.save('dactrung/spectrogram/spectrogram',data)
numpy.save('dactrung/spectrogram/target_spectrogram',target_data)
