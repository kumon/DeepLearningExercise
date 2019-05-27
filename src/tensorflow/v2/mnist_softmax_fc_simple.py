from helper import *

IMAGE_SIZE = 28 * 28
CATEGORY_NUM = 10
LEARNING_RATE = 0.1
FEATURE_DIM = 100
EPOCHS = 30
BATCH_SIZE = 100
LOG_DIR = 'log_softmax_fc'
EPS = 1e-10


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist_samples(flatten_image=True)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(FEATURE_DIM, input_shape=(IMAGE_SIZE,), activation='relu'))
    model.add(tf.keras.layers.Dense(CATEGORY_NUM, input_shape=(FEATURE_DIM,), activation='softmax'))
    model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(LEARNING_RATE), metrics=['accuracy'])

    cb = [tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)]
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=cb, validation_data=(X_test, y_test))
    print(model.evaluate(X_test, y_test))
