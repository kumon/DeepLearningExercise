import numpy as np
import tensorflow as tf


def mnist_samples(flatten_image=False, binalize_label=False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    def flatten(images):
        d, w, h = images.shape
        return images.reshape(d, w * h).astype(np.float32) / 255.0

    def binalize(labels):
        return list(map(lambda x: 1 if x == 1 else 0, labels))

    def one_hot_label(labels):
        return tf.keras.utils.to_categorical(labels, 10)

    if flatten_image:
        X_train, X_test = flatten(X_train), flatten(X_test)

    if binalize_label:
        y_train, y_test = binalize(y_train), binalize(y_test)
    else:
        y_train, y_test = one_hot_label(y_train), one_hot_label(y_test)

    return (X_train, y_train), (X_test, y_test)
