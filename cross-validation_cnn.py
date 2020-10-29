
import tensorflow as tf
#tf.enable_eager_execution()
import PIL
from PIL import Image
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

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
from scipy import stats
from sklearn.model_selection import KFold

train_image = []
y = []

input_shape = (64, 64, 1)
for i in range(1, 8):
    path = 'dataset/subject'+str(i)+'/active'
    for infile in glob.glob(os.path.join(path, '*.png')):
        img = image.load_img(infile, target_size=input_shape, grayscale=True)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(0)

    path = 'dataset/subject'+str(i)+'/inactive'
    for infile in glob.glob(os.path.join(path, '*.png')):
        #print(infile)
        img = image.load_img(infile, target_size=input_shape, grayscale=True)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(1)
    path = 'dataset/subject' + str(i) + '/medium'
    for infile in glob.glob(os.path.join(path, '*.png')):
        #print(infile)
        img = image.load_img(infile, target_size=input_shape, grayscale=True)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(2)

print(len(train_image))
X = np.array(train_image)
y = to_categorical(y)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


print('X_train: ', len(X_train))
print('X_test:  ', len(X_test))
print('y_train: ', len(y_train))
print('y_test:  ', len(y_test))

x = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

kfold = KFold(n_splits=10, shuffle=True)

batch_size = 128
num_classes = 13

epochs = 150
l = tf.keras.layers
fold_no = 1
input_shape = (128, 128, 1)
f = open("ECG_CNN_results_cv.txt", "a+")
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(x, y):

    model = tf.keras.Sequential([
        l.Conv2D(32, (3,3), padding='same', activation = 'linear', input_shape=input_shape),
        l.LeakyReLU(0.1),
        #l.MaxPooling2D((2, 2), (2, 2)),
        l.Conv2D(32, 3, padding='same', activation = 'linear'),
        l.LeakyReLU(0.1),
        l.MaxPooling2D((2, 2), (2, 2),),
        l.Dropout(0.3),
        l.Conv2D(64, 3, padding='same', activation = 'linear'),
        l.LeakyReLU(0.1),
        #l.MaxPooling2D((2, 2), (2, 2)),
        l.Conv2D(64, 3, padding='same', activation = 'linear'),
        l.LeakyReLU(0.1),
        l.MaxPooling2D((2, 2), (2, 2)),
        l.Conv2D(64, 3, padding='same', activation = 'linear'),
        l.LeakyReLU(0.1),
        l.Conv2D(64, 3, padding='same', activation = 'linear'),
        l.LeakyReLU(0.1),
        l.MaxPooling2D((2, 2), (2, 2)),
        l.Dropout(0.3),
        l.Conv2D(128, 3, padding='same', activation = 'linear'),
        l.LeakyReLU(0.1),
        #l.MaxPooling2D((2, 2), (2, 2)),
        l.Conv2D(128, 3, padding='same', activation = 'linear'),
        l.LeakyReLU(0.1),
        l.MaxPooling2D((2, 2), (2, 2)),
        l.Dropout(0.4),
        l.BatchNormalization(),
        l.Flatten(),
        l.Dense(512, activation='relu'),
        #l.Dropout(0.4),
        l.Dense(128, activation='relu'),
        l.Dropout(0.4),
        l.Dense(num_classes, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(lr=0.001)

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'])

    print('------------------------------------------------------------------------')
    print('Training for fold ',fold_no,' ...')

    model.fit(x[train], y[train],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              #validation_data=(X_test, y_test)
              validation_split=0.2
              )

    scores = model.evaluate(x[test], y[test], verbose=0)
    print('Score for fold ',fold_no,': ',model.metrics_names[0],'of ',scores[0],'; ',model.metrics_names[1],' of ', scores[1])
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    fold_no = fold_no + 1
    keras_file = 'ecg_cnn_2504.h5'
    print('Saving model to: ', keras_file)
    keras.models.save_model(model, keras_file, include_optimizer=False)

    classes=[0,1,2,3,4,5,6,7,8,9, 10, 11, 12]
    predictions = model.predict_classes(x[test])
    predictions = to_categorical(predictions)
    predictions = np.array(predictions)
    print(classification_report(y[test], predictions))

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print('> Fold ',i+1,' - Loss: ',loss_per_fold[i],' - Accuracy: ',acc_per_fold[i])
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print('> Accuracy: ',np.mean(acc_per_fold),' (+- ',np.std(acc_per_fold))
print('> Loss: ',np.mean(loss_per_fold))
print('------------------------------------------------------------------------')
