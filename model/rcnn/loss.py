import tensorflow as tf


def loss_fn(net, X, Y):
    with tf.name_scope('cross_entropy'):
        return tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=net)


def metric_fn(net, Y):
    correctly_predicted = tf.equal(Y, tf.argmax(net, axis=1))
    return tf.reduce_mean(tf.cast(correctly_predicted, tf.float32))
