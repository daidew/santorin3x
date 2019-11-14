import tensorflow as tf 

def mish(x):
    return tf.math.multiply(x, tf.math.tanh(tf.math.softplus(x)))