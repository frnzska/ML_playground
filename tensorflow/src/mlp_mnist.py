import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)


class mlp:
    """
    MLP with one hidden layer and softmax
    """
    def __init__(self, *, dims_in, dims_out, n_hidden):
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.n_hidden = n_hidden
        self.W = {'w1': tf.Variable(tf.random_normal([self.dims_in, self.n_hidden]), name='w1'),
                  'w2': tf.Variable(tf.random_normal([self.n_hidden, self.dims_out]), name='w2')}
        self.b = {'b1': tf.Variable(tf.random_normal([self.n_hidden]), name='b1'),
                  'b2': tf.Variable(tf.random_normal([self.dims_out]), name='b2')}

    @staticmethod
    def _y(*, X, W, b):
        Y_hidden = tf.nn.relu(tf.matmul(X, W['w1']) + b['b1'])
        return tf.nn.softmax(tf.matmul(Y_hidden, W['w2']) + b['b2'])


    def train(self):
        x_ph = tf.placeholder(tf.float32, [None, self.dims_in])
        y_real_ph = tf.placeholder(tf.float32, [None, self.dims_out])
        y_predict = self._y(X=x_ph, W=self.W, b=self.b)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_real_ph))
        train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

        init_op = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init_op)

        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x_ph: batch_xs, y_real_ph: batch_ys})
            if i % 100 == 0:
                print('step: ', i)
                correct_prediction = tf.equal(tf.argmax(y_real_ph, 1), tf.argmax(y_predict, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print(sess.run(accuracy, feed_dict={x_ph: mnist.test.images, y_real_ph: mnist.test.labels}))

        correct_prediction = tf.equal(tf.argmax(y_real_ph, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x_ph: mnist.test.images, y_real_ph: mnist.test.labels}))

mlp(dims_in=784, dims_out=10, n_hidden=10).train()