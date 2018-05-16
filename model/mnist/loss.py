import tensorflow as tf


def lossfn(net_out, data, labels_one_hot):
    with tf.name_scope('cross_entropy'):
        return tf.losses.sparse_softmax_cross_entropy(labels=labels_one_hot, logits=net_out)
