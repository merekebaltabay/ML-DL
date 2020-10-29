
import tensorflow as tf
#tf.enable_eager_execution()
import matplotlib.pyplot as plt
import itertools
import tempfile
import zipfile
import os, glob
import pandas as pd
import keras
from keras.preprocessing import image
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow_model_optimization.sparsity import keras as sparsity
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.applications.mobilenet import MobileNet
from keras.applications import NASNetMobile
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.nasnet import preprocess_input
from keras.models import Model
import time
from sklearn.model_selection import KFold
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, BatchNormalization, concatenate, Concatenate
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import sys
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

import PIL
from PIL import Image

train_image = []
y = []

input_shape = (64, 64, 3)

for i in range(1, 8):
    for j in range(13):
        path = 'ds_new/subject'+str(i)+'/ecg1/'+str(j)
        list_im = []
        list_im2 = []
        for infile in glob.glob(os.path.join(path, '*.png')):
            img = image.load_img(infile, target_size=input_shape)
            img = image.img_to_array(img)
            img = img/255
            list_im.append(img)
            train_image.append(img)
            y.append(j)

        path = 'ds_new/subject' + str(i) + '/ecg2/' + str(j)
        for infile in glob.glob(os.path.join(path, '*.png')):
            img = image.load_img(infile, target_size=input_shape)
            img = image.img_to_array(img)
            img = img / 255
            list_im2.append(img)
            train_image.append(img)
            y.append(j)


print(len(train_image))
X = np.array(train_image)
y = to_categorical(y)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


model1 = Sequential()
model1.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(64, 64, 3)))
#model1.add(LeakyReLU(0.1))
# l.MaxPooling2D((2, 2), (2, 2)),
model1.add(Conv2D(32, 3, padding='same', activation='relu'))
#model1.add(LeakyReLU(0.1))
model1.add(MaxPooling2D((2, 2), (2, 2), ))
model1.add(Dropout(0.3))
model1.add(Conv2D(64, 3, padding='same', activation='relu'))
#model1.add(LeakyReLU(0.1))
# l.MaxPooling2D((2, 2), (2, 2)),
model1.add(Conv2D(64, 3, padding='same', activation='relu'))
#model1.add(LeakyReLU(0.1))
model1.add(MaxPooling2D((2, 2), (2, 2)))
model1.add(Dropout(0.4))
model1.add(Conv2D(128, 3, padding='same', activation='relu'))
#model1.add(LeakyReLU(0.1))
# l.MaxPooling2D((2, 2), (2, 2)),
model1.add(Conv2D(128, 3, padding='same', activation='relu'))
#model1.add(LeakyReLU(0.1))
model1.add(MaxPooling2D((2, 2), (2, 2)))
model1.add(Dropout(0.4))
model1.add(BatchNormalization())
model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.4))
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.4))
model1.add(Dense(13, activation='softmax'))
model1.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
history = model1.fit(X_train, y_train, epochs=25, batch_size=100, verbose=2, validation_split=0.15)
test_loss, test_acc = model1.evaluate(X_test, y_test)
print(test_acc)

def make_classifier(optimizer):

    restnet = MobileNet(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    restnet = Model(restnet.input, output=output)


    for layer in restnet.layers:
        layer.trainable = True
    restnet.summary()
    model = Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(13, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

train_time = []
test_time = []
accuracies = []

temp = []
for i in y_train:
    #print(i)
    temp.append(sum(i*[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

temp = np.asarray(temp)
class_weights = class_weight.compute_class_weight('balanced',np.unique(temp),temp)
print(class_weights)

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model2 = make_classifier(optimizer)
model2.summary()
history = model2.fit(X_train, y_train, epochs=25, batch_size=100, verbose=2, validation_split=0.15)
test_loss2, test_acc2 = model2.evaluate(X_test, y_test)
print(test_acc2)

XX = model1.input
YY = model1.layers[18].output
new_model = Model(XX, YY)
Xresult = new_model.predict(X_train)
Xtesttemp = new_model.predict(X_test)
XX2 = model2.input
YY2 = model2.layers[4].output
new_model2 = Model(XX2, YY2)

Xresult2 = new_model2.predict(X_train)
Xtesttemp2 = new_model2.predict(X_test)
row = []
row2 = []
for x in range(Xresult.shape[0]):
    #print(x)
    row.append(np.concatenate((Xresult[x], Xresult2[x])))

for x in range(Xtesttemp2.shape[0]):
    #print(x)
    row2.append(np.concatenate((Xtesttemp[x], Xtesttemp2[x])))
X_train=np.asarray(row)
X_test=np.asarray(row2)

model3 = Sequential()
model3.add(Dense(128, activation='relu', input_dim=269))
model3.add(Dropout(0.3))
model3.add(Dense(64, activation='relu'))
model3.add(Dropout(0.3))
model3.add(Dense(13, activation='softmax'))
model3.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
model3.summary()
start_time = time.time()
history = model3.fit(X_train, y_train, epochs=10, batch_size=100, verbose=2, validation_split=0.15)
train_t = (time.time() - start_time)


history_dict = history.history
valid_values = history_dict['val_accuracy']
acc_values = history_dict['accuracy']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, valid_values, 'b', label='Validation accuracy', color = 'red')
plt.plot(epochs, acc_values, 'b', label='Training accuracy')
plt.title('Training loss vs accuracy')
plt.xlabel('Epochs')
plt.ylabel(''
           'Loss/Accuracy')
plt.legend()
plt.show()
start_time = time.time()
test_loss, test_acc = model3.evaluate(X_test, y_test)
test_t = (time.time() - start_time)
train_time.append(float(train_t))
test_time.append(float(test_t))
print(test_acc)


accuracies.append(test_acc)
predictions = model3.predict(X_test)
matrix = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(matrix)
print(sum(train_time) / len(train_time))
print(sum(test_time) / len(test_time))
print(sum(accuracies) / len(accuracies))


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):


    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#names = [0, 1, 2]
plot_confusion_matrix(matrix, names, title = 'MobileNet_merged')
print(sum(train_time) / len(train_time))
print(sum(test_time) / len(test_time))
print(sum(accuracies) / len(accuracies))


y_test.sum(axis=0)

y_train.sum(axis=0)


process = psutil.Process(os.getpid())
print(process.memory_info().rss)  # in bytes

model3.save("MobileNet50w_combined_12.h5")
