import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)

class PerceptronSoftmax:

    def __init__(self, *, dims_in, dims_out):
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.W = tf.Variable(tf.zeros([self.dims_in, self.dims_out]))
        self.b = tf.Variable(tf.zeros([self.dims_out]))

    @staticmethod
    def _y(*, X, W, b):
        return tf.nn.softmax(tf.matmul(X, W) + b)

    def train(self):
        # definition of variables and cost
        x_ph = tf.placeholder(tf.float32, [None, self.dims_in])
        y_real_ph = tf.placeholder(tf.float32, [None, self.dims_out])
        y_predict = self._y(X=x_ph, W=self.W, b=self.b)
        #cost = tf.reduce_mean(-tf.reduce_sum(y_real_ph * tf.log(y_predict), reduction_indices=[1])) # cross entropy
        #train_step = tf.train.GradientDescentOptimizer(0.04).minimize(cost)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_real_ph))
        train_step = tf.train.AdamOptimizer(0.05).minimize(cost)
        # run
        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)

        for i in range(2000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x_ph: batch_xs, y_real_ph: batch_ys})
            if i % 100 == 0:
                print('step: ', i)

        correct_prediction = tf.equal(tf.argmax(y_real_ph, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x_ph: mnist.test.images, y_real_ph: mnist.test.labels}))



PerceptronSoftmax(dims_in=784, dims_out=10).train()


