
import tensorflow as tf
import time
from PIL import Image
from keras.preprocessing import image
import numpy as np
from tensorflow import keras
from tensorflow_model_optimization.sparsity import keras as sparsity
import os
input_shape = (128, 128, 3)
def load_image( infilename ):
    img = image.load_img(infilename, target_size=input_shape)
    img = image.img_to_array(img)
    img = img / 255
    data = np.array(img)
    return data.reshape((1,128, 128, 3))


interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()




inference_time = []
for number in range(10):
    img = load_image('real_data/running_ecg_96_' + str(number) + '.png')

    start_time = time.time()
    #print(model.predict_classes(img))
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    tflite_prediction = np.argmax(interpreter.get_tensor(output_details[0]['index']))
    print(tflite_prediction)
    inference_time.append(time.time() - start_time)


print(sum(inference_time)/20)
