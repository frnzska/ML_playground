import tensorflow as tf
import os
cwd, data_path = os.getcwd(), 'tensorflow/data/MNIST_data/'
mnist_path = f'{cwd}/{data_path}'
print(mnist_path)
import logging
from typing import List
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(mnist_path, one_hot=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class CNN:

    def __init__(self, *, image_height: int, image_width: int, dims_out: int,
                 n_filters_per_layer: List[int], filter_sizes: List[int], pooling_sizes: List[int],
                 learning_rate: float=0.0001):
        """
        
        :param image_height: input image height
        :param image_width: input image width
        :param dims_out:  output number of dimension, n classes
        :param n_filters_per_layer: list of numbers of convolution filters per layer, conv output layers
        :param filter_sizes: size in height and with of one conv window [5, 3] means 5 x 5 kernel in first layer, 3 x 3 
                in second layer
        :param pooling_sizes: pooling window sizes, [2, 3] means 2 x 2 window in first layer, second 3 x 3 
        :param learning_rate: learning rate
    
        """
        self.__image_height = image_height
        self.__image_width = image_width
        self.__dims_out = dims_out
        self.__n_filters = n_filters_per_layer
        self.__layers = len(self.__n_filters)
        self.__filter_sizes = filter_sizes
        self.__channels = [1] + self.n_filters[:-1]
        self.__pooling_sizes = pooling_sizes
        self.learning_rate = learning_rate
        self.x_ph = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_real_ph = tf.placeholder(tf.float32, [None, self.dims_out])
        self.drop_out = tf.placeholder(tf.float32)
        self.W = None
        self.b = None
        self.n_hidden_dense = 1024
        assert len(self.__n_filters) == len(self.__filter_sizes) == len(self.__channels)\
               == len(self.__pooling_sizes) == self.__layers
        logger.info('Initialising ...')

    def _connect_conv_layers(self):
        """ Building up CNN weights from properties"""
        logger.info('... connecting the dots with ...')
        W, b = {}, {}
        i_name = [f'{i}' for i in range(0, self.layers)]

        for i, filter_size, n_filter, pooling_size, channels in zip(i_name, self.filter_sizes, self.n_filters,
                                                                    self.pooling_sizes, self.__channels):
            W[f'w{i}'] = self.weight_variable([filter_size, filter_size, channels, n_filter])
            b[f'b{i}'] = self.bias_variable([n_filter])
        self.b = b
        self.W = W
        logger.info(self.__str__())

    @staticmethod
    def conv(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def pool(x, pool_size):
        return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                              strides=[1, pool_size, pool_size, 1], padding='SAME')


    def _y(self):
        """Set desired output computation"""
        logger.info('Configure computation ..')
        h = tf.reshape(self.x_ph, [-1, self.image_width, self.image_height, 1])
        for w, _b, pool_size in zip(self.W, self.b, self.pooling_sizes):
            logger.info('Convolution and pooling')
            result_conv = tf.nn.relu(self.conv(h, self.W[w])+ self.b[_b])
            result_pool = self.pool(result_conv, pool_size)
            h = result_pool

        logger.info('dense layer')
        n_input_dense = int(h.shape[1] * h.shape[2] * h.shape[3])
        _W_dense = self.weight_variable([n_input_dense, self.n_hidden_dense])
        _b_dense = self.bias_variable([self.n_hidden_dense])
        h_flat = tf.reshape(h, [-1, n_input_dense])
        h_result = tf.nn.relu(tf.matmul(h_flat, _W_dense) + _b_dense)

        logger.info('drop out')
        h_drop = tf.nn.dropout(h_result, self.drop_out)

        logger.info('classification')
        _W_out = self.weight_variable([self.n_hidden_dense, self.dims_out])
        _b_out = self.bias_variable([self.dims_out])
        y = tf.matmul(h_drop, _W_out) + _b_out
        return y


    def train(self):
        y_predict = self._y()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_real_ph, logits=y_predict))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_real_ph, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(1000):
                batch_x, batch_y = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={self.x_ph: batch_x, self.y_real_ph: batch_y, self.drop_out: 1.0})
                if i % 50 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={self.x_ph: batch_x, self.y_real_ph: batch_y,
                                                              self.drop_out: 1.0})
                    print(f'acc taining: {train_accuracy}')

            test_acc = sess.run(accuracy, feed_dict={self.x_ph: mnist.test.images, self.y_real_ph: mnist.test.labels,
                                            self.drop_out: 1.0})
        logger.info(f'final acc: {test_acc}')
        return test_acc

    @property
    def image_height(self):
        return self.__image_height

    @property
    def image_width(self):
        return self.__image_width

    @property
    def dims_out(self):
        return self.__dims_out

    @property
    def n_filters(self):
        return self.__n_filters

    @n_filters.setter
    def n_filters(self, val: List[int]):
        self.__n_filters = val

    @property
    def layers(self):
        return self.__layers

    @property
    def filter_sizes(self):
        return self.__filter_sizes

    @filter_sizes.setter
    def filter_sizes(self, val: List[int]):
        self.__filter_sizes = val

    @property
    def pooling_sizes(self):
        return self.__pooling_sizes

    @pooling_sizes.setter
    def pooling_sizes(self, val):
        self.__pooling_sizes = val

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


#a = CNN(image_height=28, image_width=28, dims_out=10, n_filters_per_layer= [30,60], filter_sizes=[5,5], pooling_sizes=[2,2], learning_rate=0.0001)
#a._connect_conv_layers()
#a._y()
#a.train()