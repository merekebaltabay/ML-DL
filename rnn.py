#requires tensorflow > 1.4 and keras v>2
from __future__ import print_function

from matplotlib import pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
#import coremltools
import tempfile
from scipy import stats
from sklearn.model_selection import train_test_split
#from IPython.display import display, HTML
import tensorflow as tf
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, LSTM, Embedding, SimpleRNN, GRU
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
file_path = 'mHealth_subjects.csv'

names=['accX', 'accY', 'accZ', 'ecg1', 'ecg2', 'alaX', 'alaY', 'alaZ', 'glaX', 'glaY', 'glaZ', 'mlaX', 'mlaY', 'mlaZ',  'arlaX', 'arlaY', 'arlaZ', 'grlaX', 'grlaY', 'grlaZ', 'mrlaX', 'mrlaY', 'mrlaZ', 'label']
df = pd.read_csv(file_path,
 #               usecols=[0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 23],
                 names=names,
                skiprows=1,
                delimiter=';')
#print(df)
#df.dropna(axis=0, how='any', inplace=True)
print(df.shape)
print(df.shape)

TIME_PERIODS = 80
STEP_DISTANCE = 40
print(df.label.value_counts())


df[df['label']==0] = df[df['label']==0].sample(15360).astype(int)
df.dropna(axis=0, how='any', inplace=True)
print(df.shape)
print(df.label.value_counts())

for n in names:
    if(n!= 'label'):
        df[n] = df[n]/df[n].max()


pd.options.mode.chained_assignment = None

def create_segments_and_labels(df, time_steps, step):
    N_FEATURES = 23
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        accX = df['accX'].values[i: i + time_steps]
        accY = df['accY'].values[i: i + time_steps]
        accZ = df['accZ'].values[i: i + time_steps]
        ecg1 = df['ecg1'].values[i: i + time_steps]
        ecg2 = df['ecg2'].values[i: i + time_steps]
        alaX = df['alaX'].values[i: i + time_steps]
        alaY = df['alaY'].values[i: i + time_steps]
        alaZ = df['alaZ'].values[i: i + time_steps]
        glaX = df['glaX'].values[i: i + time_steps]
        glaY = df['glaY'].values[i: i + time_steps]
        glaZ = df['glaZ'].values[i: i + time_steps]
        mlaX = df['mlaX'].values[i: i + time_steps]
        mlaY = df['mlaY'].values[i: i + time_steps]
        mlaZ = df['mlaZ'].values[i: i + time_steps]

        arlaX = df['arlaX'].values[i: i + time_steps]
        arlaY = df['arlaY'].values[i: i + time_steps]
        arlaZ = df['arlaZ'].values[i: i + time_steps]
        grlaX = df['grlaX'].values[i: i + time_steps]
        grlaY = df['grlaY'].values[i: i + time_steps]
        grlaZ = df['grlaZ'].values[i: i + time_steps]

        mrlaX = df['mrlaX'].values[i: i + time_steps]
        mrlaY = df['mrlaY'].values[i: i + time_steps]
        mrlaZ = df['mrlaZ'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + time_steps])[0][0]


        segments.append([accX, accY, accZ, ecg1, ecg2, alaX, alaY, alaZ, glaX, glaY, glaZ, mlaX, mlaY, mlaZ, arlaX, arlaY, arlaZ, grlaX, grlaY, grlaZ, mrlaX, mrlaY, mrlaZ])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

df, y = create_segments_and_labels(df, TIME_PERIODS, STEP_DISTANCE)

x_train, x_test, y_train, y_test = train_test_split(df, y,  test_size=0.2, random_state=42)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)
print(x_train.shape)
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
input_shape = (num_time_periods*num_sensors)

x_train = x_train.reshape(x_train.shape[0], input_shape)
x_test = x_test.reshape(x_test.shape[0], input_shape)

#print('x_train shape:', x_train.shape)
#print('input_shape:', input_shape)
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

y_train= to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_train.shape[1]
#print('New y_train shape: ', y_train.shape)

model = Sequential()

model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(GRU(256, activation='relu', return_sequences=True))
#model.add(SimpleRNN(128))
model.add(LSTM(128))

model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
#model_m.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())


opt = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 128
EPOCHS = 150

history = model.fit(x_train, y_train,
                    #validation_data=(x_test, y_test),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=1)


score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
f = open("ECG_RNN_results_11_04.txt", "a+")
f.write('Test loss:'+str(score[0]))
f.write('Test accuracy:'+str(score[1]))
f.close()

Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(x_test)
print(classification_report(Y_test, y_pred))


keras_file = 'ecg_rnn.h5'
print('Saving model to: ', keras_file)
keras.models.save_model(model, keras_file, include_optimizer=False)

labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
fig = plt.figure()

species = np.array(y_test).argmax(axis=0)
predictions = np.array(y_pred).argmax(axis=0)
confusion_matrix(y_test, y_pred)

plot_confusion_matrix(model, x_test, y_test,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize="true")
plt.show()
fig.savefig("conf_matrix_rnn.png")
