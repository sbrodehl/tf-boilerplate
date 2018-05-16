import tensorflow as tf


# Define data loaders #####################################
# See https://gist.github.com/peterroelants/9956ec93a07ca4e9ba5bc415b014bcca
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self, func=None):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = func

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)