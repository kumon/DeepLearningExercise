import numpy as np
import tensorflow as tf
from scipy import spatial
import operator

def weight_variable(shape, name=None):
    initial = tf.glorot_uniform_initializer()
    return tf.Variable(initial(shape), name=name)

def bias_variable(shape, name=''):
    initial = tf.zeros_initializer()
    return tf.Variable(initial(shape), name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def prewitt_filter():
    v = np.array([[ 1, 0, -1]] * 3)
    h = v.swapaxes(0, 1)
    f = np.zeros(3 * 3 * 1 * 2).reshape(3, 3, 1, 2)
    f[:, :, 0, 0] = v
    f[:, :, 0, 1] = h
    return tf.constant(f, dtype = tf.float32, name='prewitt')

def mnist_samples(flatten_image=False, binalize_label=False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    def flatten(images):
        d, w, h = images.shape
        return images.reshape(d, w * h).astype(np.float32) / 255.0

    def binalize(labels):
        return list(map(lambda x: [1] if x == 1 else [0], labels))

    def one_hot_label(labels):
        return tf.keras.utils.to_categorical(labels, 10)

    if flatten_image:
        X_train, X_test = flatten(X_train), flatten(X_test)

    if binalize_label:
        y_train, y_test = binalize(y_train), binalize(y_test)
    else:
        y_train, y_test = one_hot_label(y_train), one_hot_label(y_test)

    return (X_train, y_train), (X_test, y_test)
