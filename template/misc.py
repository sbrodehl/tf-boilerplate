import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunArgs


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


class OneTimeSummarySaverHook(tf.train.SummarySaverHook):
    """One-Time SummarySaver
    Saves summaries every N steps.

    E.g. can be used for saving the source code as text.
    """

    def __init__(self, output_dir=None, summary_writer=None, scaffold=None, summary_op=None):
        self._summary_op = summary_op
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._scaffold = scaffold

        class emptytimer():
            def update_last_triggered_step(*args,**kwargs):
                pass
        self._timer = emptytimer()

    def begin(self):
        super().begin()
        self._done = False

    def before_run(self, run_context):	# pylint: disable=unused-argument
        self._request_summary = not self._done
        requests = {"global_step": self._global_step_tensor}
        if self._request_summary:
            if self._get_summary_op() is not None:
                # print(self._iter_count)
                requests["summary"] = self._get_summary_op()

        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        super().after_run(run_context,run_values)
        self._done = True


def ExperimentTemplate() -> str:
    """A template with Markdown syntax.

    :return: str with Markdown template
    """
    return """
Experiment
==========

Any [markdown code](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) can be used to describe this experiment.
For instance, you can find the automatically generated used settings of this run below.


Current Settings
----------------

| Argument | Value |
| -------- | ----- |
"""
