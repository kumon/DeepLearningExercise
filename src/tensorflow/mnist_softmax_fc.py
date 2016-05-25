from helper import *

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
CATEGORY_NUM = 10
LEARNING_RATE = 0.1
FEATURE_DIM = 100
TRAINING_LOOP = 20000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_softmax_fc'
SUMMARY_INTERVAL = 100

mnist = input_data.read_data_sets('data', one_hot=True)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        y_ = tf.placeholder(tf.float32, [None, CATEGORY_NUM], name='labels')
        x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='input_images')

    with tf.name_scope('fully-connected'):
        W_fc = weight_variable([IMAGE_SIZE, FEATURE_DIM], name='weight_fc')
        b_fc = bias_variable([FEATURE_DIM], name='bias_fc')
        h_fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc)

    with tf.name_scope('readout'):
        W = weight_variable([FEATURE_DIM, CATEGORY_NUM], name='weight')
        b = bias_variable([CATEGORY_NUM], name='bias')
        y = tf.nn.softmax(tf.matmul(h_fc, W) + b)

    with tf.name_scope('optimize'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(SUMMARY_DIR + '/test', sess.graph)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy_summary = tf.scalar_summary('accuracy', accuracy)
        test_accuracy_summary = tf.scalar_summary('accuracy', accuracy)

        sess.run(tf.initialize_all_variables())
        for i in range(TRAINING_LOOP + 1):
            batch = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, {x: batch[0], y_: batch[1]})

            if i % SUMMARY_INTERVAL == 0:
                print('step %d' % i)
                summary = sess.run(tf.merge_summary([train_accuracy_summary]), {x: batch[0], y_: batch[1]})
                train_writer.add_summary(summary, i)
                summary = sess.run(tf.merge_summary([test_accuracy_summary]), {x: mnist.test.images, y_: mnist.test.labels})
                test_writer.add_summary(summary, i)
