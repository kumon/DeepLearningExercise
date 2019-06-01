from helper import *

IMAGE_WIDTH, IMAGE_HEIGHT = 28, 28
CATEGORY_NUM = 10
LEARNING_RATE = 0.1
FILTER_NUM = 2
FEATURE_DIM = 100
EPOCHS = 15
BATCH_SIZE = 100
LOG_DIR = 'log_fixed_cnn_pl'


class Prewitt(tf.keras.layers.Layer):
    def build(self, input_shape):
        v = np.array([[ 1, 0, -1]] * 3)
        h = v.swapaxes(0, 1)
        self.kernel = tf.constant(np.dstack([v, h]).reshape((3, 3, 1, 2)), dtype = tf.float32, name='prewitt')
        self.built = True

    def call(self, x):
        x_ = tf.reshape(x, [-1, x.shape[1], x.shape[2], 1])
        return tf.abs(tf.nn.conv2d(x_, self.kernel, strides=[1, 1, 1, 1], padding='SAME'))

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist_samples()

    model = tf.keras.models.Sequential()
    model.add(Prewitt((IMAGE_HEIGHT * IMAGE_WIDTH, FILTER_NUM), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(FEATURE_DIM, activation='relu'))
    model.add(tf.keras.layers.Dense(CATEGORY_NUM, activation='softmax'))
    model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(LEARNING_RATE), metrics=['accuracy'])

    cb = [tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)]
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=cb, validation_data=(X_test, y_test))
    print(model.evaluate(X_test, y_test))

