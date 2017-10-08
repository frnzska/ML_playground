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

