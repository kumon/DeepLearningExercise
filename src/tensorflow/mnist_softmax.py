from helper import *

IMAGE_SIZE = 28 * 28
CATEGORY_NUM = 10
LEARNING_RATE = 0.1
TRAINING_LOOP = 20000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_softmax'
SUMMARY_INTERVAL = 1000
BUFFER_SIZE = 1000
EPS = 1e-10

with tf.Graph().as_default():
    (X_train, y_train), (X_test, y_test) = mnist_samples(flatten_image=True)
    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(int(TRAINING_LOOP * BATCH_SIZE / X_train.shape[0]) + 1)
    next_batch = ds.make_one_shot_iterator().get_next()

    with tf.name_scope('input'):
        y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM], name='labels')
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='input_images')

    with tf.name_scope('readout'):
        W = weight_variable([IMAGE_SIZE, CATEGORY_NUM], name='weight')
        b = bias_variable([CATEGORY_NUM], name='bias')
        y = tf.nn.softmax(tf.matmul(x, W) + b)

    with tf.name_scope('optimize'):
        y = tf.clip_by_value(y, EPS, 1.0)
        cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
        cross_entropy_summary = tf.summary.scalar('cross entropy', cross_entropy)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test')

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_LOOP + 1):
            images, labels = sess.run(next_batch)
            sess.run(train_step, {x: images, y_: labels})

            if i % SUMMARY_INTERVAL == 0:
                train_acc, summary = sess.run(
                        [accuracy, tf.summary.merge([cross_entropy_summary, accuracy_summary])],
                        {x: images, y_: labels})
                train_writer.add_summary(summary, i)
                test_acc, summary = sess.run(
                        [accuracy, tf.summary.merge([cross_entropy_summary, accuracy_summary])],
                        {x: X_test, y_: y_test})
                test_writer.add_summary(summary, i)
                print(f'step: {i}, train-acc: {train_acc}, test-acc: {test_acc}')
