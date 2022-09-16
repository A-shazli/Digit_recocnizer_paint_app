from keras.datasets import mnist
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = preprocessing.normalize(x_test)
plt.imshow(x_test[21])
plt.show()
# x_test[100] is a 6
model = load_model('mnist.h5')
w = x_test[21].reshape(1,28,28,1)
pred = model.predict(w)[0]
pred = tf.nn.softmax(pred)
pred = pred.numpy() * 100
print(np.argmax(pred))
print(max(pred))