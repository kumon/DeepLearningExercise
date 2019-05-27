from helper import *

IMAGE_SIZE = 28 * 28
CATEGORY_NUM = 10
LEARNING_RATE = 0.1
FEATURE_DIM = 100
EPOCHS = 30
BATCH_SIZE = 100
LOG_DIR = 'log_softmax_fc'
EPS = 1e-10


def loss_fn(y_true, y):
    y = tf.clip_by_value(y, EPS, 1.0)
    return -tf.reduce_sum(y_true * tf.math.log(y), axis=1)

class Dense(tf.keras.layers.Layer):
    def __init__(self, units, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.W = self.add_weight(
            name='weight',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform()
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros()
        )
        self.built = True

    def call(self, x):
        if self.activation is None:
            raise Exception('Activation function is None')
        return self.activation(tf.matmul(x, self.W) + self.b)

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist_samples(flatten_image=True)

    model = tf.keras.models.Sequential()
    model.add(Dense(FEATURE_DIM, input_shape=(IMAGE_SIZE,), activation='relu'))
    model.add(Dense(CATEGORY_NUM, input_shape=(FEATURE_DIM,), activation='softmax'))
    model.compile(loss=loss_fn, optimizer=tf.keras.optimizers.SGD(LEARNING_RATE), metrics=['accuracy'])

    cb = [tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)]
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=cb, validation_data=(X_test, y_test))
    print(model.evaluate(X_test, y_test))

