import numpy as np
import tensorflow as tf
import os

# quell CPU instruction warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# buffer size for shuffling data
buf_size = 10000
num_epochs = 3

def parse_func(l):
    record_defaults = [[0], [0], [0], [0]]
    features = tf.decode_csv(l, field_delim=' ',
        use_quote_delim=False, record_defaults=record_defaults)
    return tf.stack(features)

dataset = tf.data.TextLineDataset('um/all.dta')
dataset = dataset.map(parse_func)
dataset = dataset.shuffle(buffer_size=buf_size)
dataset = dataset.repeat(num_epochs)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            val = sess.run(next_element)
            
        except tf.errors.OutOfRangeError:
            break
