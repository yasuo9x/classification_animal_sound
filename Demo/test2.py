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
from sklearn import neighbors,datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x = numpy.array([[1,2,3,4],[9,10,11,12]])
y = numpy.array([[[1,2,3,4]],[[1,7,9,10]],[[5,5,6,7]],[[9,9,9,9]],[[9,10,11,12]]])
target_y = numpy.array([0,1,2,3])


# clf = neighbors.KNeighborsClassifier(n_neighbors=10,p=2,weights='distance')
# clf.fit(y,target_y)
# y = clf.kneighbors(x)
# y_pred = clf.predict(x) # du doan lay nhan cua data moi
print(x)