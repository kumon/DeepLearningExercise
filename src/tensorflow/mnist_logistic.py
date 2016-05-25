from helper import *

IMAGE_SIZE = 28 * 28
CATEGORY_NUM = 1
LEARNING_RATE = 0.1
TRAINING_LOOP = 20000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_logistic'
SUMMARY_INTERVAL = 100

mnist = input_data.read_data_sets('data', one_hot=True)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM], name='labels')
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='input_images')

    with tf.name_scope('readout'):
        W = weight_variable([IMAGE_SIZE, CATEGORY_NUM], name='weight')
        b = bias_variable([CATEGORY_NUM], name='bias')
        y = tf.nn.sigmoid(tf.matmul(x, W) + b)

    with tf.name_scope('optimize'):
        log_likelihood = tf.reduce_mean(tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(-log_likelihood)
        log_likelihood_summary = tf.scalar_summary('log likelihood', log_likelihood)

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/test', sess.graph)

        correct_prediction = tf.equal(tf.sign(y - 0.5), tf.sign(y_ - 0.5))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy_summary = tf.scalar_summary('accuracy', accuracy)
        test_accuracy_summary = tf.scalar_summary('accuracy', accuracy)

        sess.run(tf.initialize_all_variables())
        for i in range(TRAINING_LOOP + 1):
            images, labels = mnist.train.next_batch(BATCH_SIZE)
            labels = [[1] if l[0] == 1 else [0] for l in labels]
            sess.run(train_step, {x: images, y_: labels})

            if i % SUMMARY_INTERVAL == 0:
                print('step %d' % i)
                summary = sess.run(tf.merge_summary([log_likelihood_summary, train_accuracy_summary]), {x: images, y_: labels})
                train_writer.add_summary(summary, i)
                summary = sess.run(tf.merge_summary([test_accuracy_summary]), {x: mnist.test.images, y_: [[1] if l[0] == 1 else [0] for l in mnist.test.labels]})
                test_writer.add_summary(summary, i)
