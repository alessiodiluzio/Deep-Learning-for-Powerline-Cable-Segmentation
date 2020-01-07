import tensorflow as tf


def temp_filter(batch):
    mean = 0
    for img in batch:
        mean += img
    mean /= len(batch)
    mean = tf.where(tf.math.greater(mean, 0.5), tf.ones_like(mean), 0)
    return mean
