from helper import *

IMAGE_SIZE = 28 * 28
CATEGORY_NUM = 1
LEARNING_RATE = 0.1
TRAINING_LOOP = 20000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_logistic'
SUMMARY_INTERVAL = 1000
BUFFER_SIZE = 1000
EPS = 1e-10

with tf.Graph().as_default():
    mnist_train, mnist_test = mnist_samples(binalize=True)
    n_train = mnist_train[0].shape[0]
    ds = tf.data.Dataset.from_tensor_slices(mnist_train)
    ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(int(TRAINING_LOOP * BATCH_SIZE / n_train) + 1)
    next_batch = ds.make_one_shot_iterator().get_next()

    with tf.name_scope('input'):
        y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM], name='labels')
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='input_images')

    with tf.name_scope('readout'):
        W = weight_variable([IMAGE_SIZE, CATEGORY_NUM], name='weight')
        b = bias_variable([CATEGORY_NUM], name='bias')
        #y = tf.nn.sigmoid(tf.matmul(x, W) + b)
        y = tf.clip_by_value(tf.nn.sigmoid(tf.matmul(x, W) + b), EPS, 1.0 - EPS)

    with tf.name_scope('optimize'):
        log_likelihood = tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(-log_likelihood)
        log_likelihood_summary = tf.summary.scalar('log likelihood', log_likelihood)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test')

        correct_prediction = tf.equal(tf.sign(y - 0.5), tf.sign(y_ - 0.5))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_LOOP + 1):
            images, labels = sess.run(next_batch)
            sess.run(train_step, {x: images, y_: labels})

            if i % SUMMARY_INTERVAL == 0:
                print('step %d' % i)
                summary = sess.run(
                        tf.summary.merge([log_likelihood_summary, accuracy_summary]),
                        {x: images, y_: labels})
                train_writer.add_summary(summary, i)
                summary = sess.run(
                        tf.summary.merge([accuracy_summary]),
                        {x: mnist_test[0], y_: mnist_test[1]})
                test_writer.add_summary(summary, i)
