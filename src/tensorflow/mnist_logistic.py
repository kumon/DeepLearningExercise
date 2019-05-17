from helper import *

IMAGE_SIZE = 28 * 28
CATEGORY_NUM = 1
LEARNING_RATE = 0.1
TRAINING_LOOP = 20000
BATCH_SIZE = 100
SUMMARY_DIR = 'log_logistic'
SUMMARY_INTERVAL = 1000
EPS = 1e-10

mnist = input_data.read_data_sets('data', one_hot=True)

def gen_label(labels):
    return list(map(lambda x: [1] if x[0] == 1 else [0], labels))

with tf.Graph().as_default():
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
            images, org_labels = mnist.train.next_batch(BATCH_SIZE)
            labels = gen_label(org_labels)
            sess.run(train_step, {x: images, y_: labels})

            if i % SUMMARY_INTERVAL == 0:
                print('step %d' % i)
                summary = sess.run(
                        tf.summary.merge([log_likelihood_summary, accuracy_summary]),
                        {x: images, y_: labels})
                train_writer.add_summary(summary, i)
                summary = sess.run(
                        tf.summary.merge([accuracy_summary]),
                        {x: mnist.test.images, y_: gen_label(mnist.test.labels)})
                test_writer.add_summary(summary, i)
