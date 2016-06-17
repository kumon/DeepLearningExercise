from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf
from scipy import spatial
import operator

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=''):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

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

