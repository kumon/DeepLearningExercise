from helper import *

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH = 28, 28, 1
CATEGORY_NUM = 10
LEARNING_RATE = 0.1
FILTER_SIZE = 5
FILTER_NUM = 32
FEATURE_DIM = 100
KEEP_PROB = 0.5
EPOCHS = 15
BATCH_SIZE = 100
LOG_DIR = 'log_cnn_sl'


if __name__ == '__main__':
    sh = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)
    (X_train, y_train), (X_test, y_test) = mnist_samples(shape=sh)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(FILTER_NUM, (FILTER_SIZE, FILTER_SIZE), input_shape=sh))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(FEATURE_DIM, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=1-KEEP_PROB))
    model.add(tf.keras.layers.Dense(CATEGORY_NUM, activation='softmax'))
    model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.SGD(LEARNING_RATE), metrics=['accuracy'])

    cb = [tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)]
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=cb, validation_data=(X_test, y_test))
    print(model.evaluate(X_test, y_test))

