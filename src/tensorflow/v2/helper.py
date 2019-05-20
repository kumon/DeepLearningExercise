import numpy as np
import tensorflow as tf


def mnist_samples(binalize=False):
    train, test = tf.keras.datasets.mnist.load_data()

    def preprocess(images, labels):
        d, w, h = images.shape
        return (images.reshape(d, w * h).astype(np.float32) / 255.0, labels)

    def binalize_label(labels):
        return list(map(lambda x: 1 if x == 1 else 0, labels))

    return preprocess(train[0], binalize_label(train[1])), preprocess(test[0], binalize_label(test[1]))
