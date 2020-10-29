#requires tensorflow > 1.4 and keras v>2
from __future__ import print_function

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import tempfile
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, LSTM
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import ecg_plot


for k in range(9, 11):
    file_path = 'MHEALTHDATASET/mHealth_subject'+str(k)+'.csv'


    names=[ 'ecg1', 'ecg2', 'label']
    df = pd.read_csv(file_path,
                     usecols=[ 3, 4, 23],
                     names=names,
                    skiprows=1,
                    delimiter=';')


    print(df.label.value_counts())

    df.dropna(axis=0, how='any', inplace=True)
    print(df.shape)
    #df = df[df.label != 0]
    print(df.shape)

    TIME_PERIODS = 96
    STEP_DISTANCE = 12
    '''
    for n in names:
        if(n!= 'label'):
            #print (n)
            df[n] = df[n]/df[n].max()
    '''


    def create_segments_and_labels(df, time_steps, step):
        N_FEATURES = 2
        segments = []
        labels = []
        for i in range(0, len(df) - time_steps, step):
            ecg1 = df['ecg1'].values[i: i + time_steps]
            ecg2 = df['ecg2'].values[i: i + time_steps]

            label = stats.mode(df['label'][i: i + time_steps])[0][0]
            ecg_plot.plot((ecg1, ecg2), sample_rate=50, title = '',  show_grid= False, show_lead_name = False)
            ecg_plot.save_as_png("mhealth/subject"+str(k)+"/ecg/"+str(label)+"/img_"+str(i))
            #ecg_plot.show()
            segments.append([ecg1, ecg2])
            labels.append(label)

            # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)

        return reshaped_segments, labels

    df, y = create_segments_and_labels(df, TIME_PERIODS, STEP_DISTANCE)