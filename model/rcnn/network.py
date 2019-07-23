import tensorflow as tf
import tensorflow.contrib.layers as tfl

"""Copied from the almighty Christian Hundt;

CECAM/CSM/IRTG School 2018: Machine Learning in Scientific Computing
https://github.com/CECAML/school_nierstein_2018/blob/master/Convnet%20TF.ipynb
"""


def prelu(net):
    alpha = tf.Variable(0.0, dtype=net.dtype)
    return tf.maximum(alpha * net, net)


def residual_conv_block(net, num_filters, kernel_size, stride, is_training=True):
    # let us cache the input tensor and downsample it
    inp = tfl.avg_pool2d(net, kernel_size, stride, padding="SAME")

    # now convolve with stride (potential downsampling)
    net = tfl.conv2d(net, num_filters, kernel_size, stride, activation_fn=tf.identity, padding="SAME")

    # normalize the output
    net = tfl.batch_norm(net, is_training=is_training, activation_fn=tf.identity)

    # now convolve again but do not downsample
    net = tfl.conv2d(net, num_filters, kernel_size, stride=1, activation_fn=tf.identity, padding="SAME")

    return prelu(tf.concat((net, inp), axis=-1))


def network(X, Y, mode=tf.estimator.ModeKeys.TRAIN, classes=None):
    net = tf.identity(X)

    net = residual_conv_block(net, 16, 3, 2)
    net = residual_conv_block(net, 32, 3, 2)
    net = residual_conv_block(net, 64, 3, 2)
    net = residual_conv_block(net, 128, 3, 2)

    net = tf.reduce_mean(net, axis=(1, 2))
    net = tfl.fully_connected(net, 10, activation_fn=tf.identity)

    return net
