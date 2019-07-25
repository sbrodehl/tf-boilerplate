import tensorflow as tf

"""Copied from the almighty Christian Hundt;

CECAM/CSM/IRTG School 2018: Machine Learning in Scientific Computing
https://github.com/CECAML/school_nierstein_2018/blob/master/Convnet%20TF.ipynb
"""


def prelu(net):
    alpha = tf.Variable(0.0, dtype=net.dtype)
    return tf.maximum(alpha * net, net)


def residual_conv_block(net, num_filters, kernel_size, stride):
    layers = tf.keras.layers

    # let us cache the input tensor and downsample it
    inp = layers.AveragePooling2D(kernel_size, stride, padding="SAME")(net)

    # now convolve with stride (potential downsampling)
    net = layers.Conv2D(num_filters, kernel_size, stride, padding="SAME")(net)

    # normalize the output
    net = layers.BatchNormalization()(net)

    # now convolve again but do not downsample
    net = layers.Conv2D(num_filters, kernel_size, stride, strides=(1, 1), padding="SAME")(net)

    return prelu(layers.concatenate((net, inp), axis=-1))


def network(x, y, mode=tf.estimator.ModeKeys.TRAIN, classes=None):
    layers = tf.keras.layers

    net = tf.identity(x)

    net = residual_conv_block(net, 16, 3, 2)
    net = residual_conv_block(net, 32, 3, 2)
    net = residual_conv_block(net, 64, 3, 2)
    net = residual_conv_block(net, 128, 3, 2)

    net = layers.GlobalAveragePooling2D()(net)
    net = layers.Dense(classes)(net)

    return net
