import os
from tensorflow.src.utils.text_processing import text_to_idx_dict
import tensorflow as tf

cwd, data_path = os.getcwd(), 'tensorflow/data/text_data/lord_of_the_rings.txt'
path = f'{cwd}/{data_path}'
raw_data = open(path).read()

words_dict = text_to_idx_dict(text=raw_data)
vocab_size = len(words_dict)

class rnn:

    def __init__(self, *, n_hidden, vocab_size, n_inputs):
        self.W = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
        self.b =  tf.Variable(tf.random_normal([vocab_size]))
        self.n_inputs = n_inputs

