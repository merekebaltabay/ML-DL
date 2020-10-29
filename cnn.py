import tensorflow as tf
import tempfile
import zipfile
import os, glob
import pandas as pd
import keras
from keras.preprocessing import image
import numpy as np
import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import itertools
from keras import models
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow_model_optimization.sparsity import keras as sparsity
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import time
train_image = []
y = []
input_shape = (64, 64, 3)

for i in range(1, 5):
    path = 'dataset/subject'+str(i)+'/active'
    for infile in glob.glob(os.path.join(path, '*.png')):
        img = image.load_img(infile, target_size=input_shape)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(0)

    path = 'dataset/subject'+str(i)+'/inactive'
    for infile in glob.glob(os.path.join(path, '*.png')):
        #print(infile)
        img = image.load_img(infile, target_size=input_shape)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(1)
    path = 'dataset/subject' + str(i) + '/medium'
    for infile in glob.glob(os.path.join(path, '*.png')):
        #print(infile)
        img = image.load_img(infile, target_size=input_shape)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)
        y.append(2)

print(len(train_image))
X = np.array(train_image)
y = to_categorical(y)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

#print(a, b, c)
#print(y)
#print('X_train: ', len(X_train))
#print('X_test:  ', len(X_test))
#print('y_train: ', len(y_train))
#print('y_test:  ', len(y_test))

batch_size = 100
num_classes = 3

epochs = 25
l = tf.keras.layers

model = tf.keras.Sequential([
    l.Conv2D(32, (3,3), padding='same', activation='linear', input_shape=input_shape),
    l.LeakyReLU(0.1),
    #l.MaxPooling2D((2, 2), (2, 2)),
    l.Conv2D(32, 3, padding='same', activation='linear'),
    l.LeakyReLU(0.1),
    l.MaxPooling2D((2, 2), (2, 2),),
    l.Dropout(0.3),
    l.Conv2D(64, 3, padding='same', activation='linear'),
    l.LeakyReLU(0.1),
    #l.MaxPooling2D((2, 2), (2, 2)),
    l.Conv2D(64, 3, padding='same', activation='linear'),
    l.LeakyReLU(0.1),
    l.MaxPooling2D((2, 2), (2, 2)),
    l.Dropout(0.4),
    l.Conv2D(128, 3, padding='same', activation='linear'),
    l.LeakyReLU(0.1),
    #l.MaxPooling2D((2, 2), (2, 2)),
    l.Conv2D(128, 3, padding='same', activation='linear'),
    l.LeakyReLU(0.1),
    l.MaxPooling2D((2, 2), (2, 2)),
    l.Dropout(0.4),
    l.BatchNormalization(),
    l.Flatten(),
    l.Dense(512, activation='relu'),
    l.Dropout(0.4),
    l.Dense(128, activation='relu'),
    l.Dropout(0.4),
    l.Dense(num_classes, activation='softmax')
])

opt = tf.keras.optimizers.Adam(lr=0.001)

train_time = []
test_time = []
accuracies = []


temp = []
for i in y_train:
    temp.append(sum(i*[0, 1, 2]))
temp = np.asarray(temp)
class_weights = class_weight.compute_class_weight('balanced',np.unique(temp),temp)
print(class_weights)



model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=opt,
    metrics=['accuracy'])

start_time = time.time()
checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                               monitor = 'val_acc',
                               verbose=1,
                               save_best_only=True)

history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[checkpointer],
          validation_split=0.15,
          class_weight=class_weights
          )
train_t = (time.time() - start_time)

history_dict = history.history
valid_values = history_dict['val_acc']
acc_values = history_dict['acc']
epochs = range(1, len(acc_values) + 1)
fig = plt.figure(figsize=(8, 8))
plt.plot(epochs, valid_values, 'b', label='Validation accuracy', color = 'red')
plt.plot(epochs, acc_values, 'b', label='Training accuracy')
plt.title('Training loss vs accuracy')
plt.xlabel('Epochs')
plt.ylabel(''
           'Loss/Accuracy')
plt.legend()
#plt.show()
fig.savefig("cnn_accuracy.png")
plt.close(fig)
start_time = time.time()
score = model.evaluate(X_test, y_test, verbose=0)
test_t = (time.time() - start_time)
train_time.append(float(train_t))
test_time.append(float(test_t))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
test_loss = score[0]
test_acc = score[1]

model.load_weights('best_weights.hdf5')
model.save('cnn.h5')


accuracies.append(test_acc)
predictions = model.predict(X_test)
names = [0, 1, 2]
plot_confusion_matrix(matrix, names, title = 'CNN')
print(sum(train_time) / len(train_time))
print(sum(test_time) / len(test_time))
print(sum(accuracies) / len(accuracies))


y_test.sum(axis=0)
y_train.sum(axis=0)


classes=[0, 1,2]
predictions = model.predict_classes(X_test)
predictions = to_categorical(predictions)
predictions = np.array(predictions)
print(classification_report(y_test, predictions))



#to plot image after layers
img_path = 'dataset/subject5/active/s1_9_ecg1_1.png'

img = image.load_img(img_path, target_size=(64, 64))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.figure(figsize = (8, 8))
plt.imshow(img_tensor[0])
plt.show()
plt.savefig("cnn_img_tensor.png")
print(img_tensor.shape)

layer_outputs = [layer.output for layer in model.layers[:12]] # Extracts the outputs of the top 12 layers
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
    n_features = layer_activation.shape[-1]  # Number of features in the feature map
    size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):  # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,  # Displays the grid
            row * size: (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()

