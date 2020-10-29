import tensorflow as tf
import tempfile
import os, glob
import pandas as pd
from tqdm import tqdm
import keras
from keras.preprocessing import image
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow_model_optimization.sparsity import keras as sparsity


train_image = []
y = []
input_shape = (228, 228, 1)

for i in range(1, 11):
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

#print(a, b, c)
#print(y)
#print('X_train: ', len(X_train))
#print('X_test:  ', len(X_test))
#print('y_train: ', len(y_train))
#print('y_test:  ', len(y_test))

batch_size = 64
num_classes = 3

epochs = 50
l = tf.keras.layers

num_train_samples = X_train.shape[0]
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs


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
    # l.MaxPooling2D((2, 2), (2, 2)),
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

callbacks = [
    sparsity.UpdatePruningStep(),
    keras.callbacks.ModelCheckpoint(
        filepath='cnn_pruned_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
]

new_pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
}

new_pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)
#new_pruned_model.summary()

new_pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=opt,
    metrics=['accuracy'])


new_pruned_model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(X_test, y_test))

score = new_pruned_model.evaluate(X_test, y_test, verbose=0)
print('New pruned Test loss:', score[0])
print('New pruned Test accuracy:', score[1])
