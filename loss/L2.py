from loss import BaseLoss

import tensorflow as tf


class L2(BaseLoss):
    """L2 Loss"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, indices):
        # check if tensor needs some love
        if indices.shape[-1].value == 1:
            indices = tf.squeeze(indices, axis=-1)
        # check if only one time stamp is used as label
        if indices.shape[1].value > 1:
            indices = indices[:, -1, ...]
            indices = tf.expand_dims(indices, axis=1)
        labels = tf.one_hot(indices, self.classes)
        l2 = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels, logits), name=str(self.__class__.__name__))
        with tf.name_scope(self.scope):
            tf.compat.v1.summary.scalar(str(self.__class__.__name__), l2)
        return l2
