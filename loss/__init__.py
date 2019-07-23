import argparse
import importlib.util
import pkgutil
import os
from pathlib import Path

import tensorflow as tf


class BaseLoss(object):
    """Loss which computes things."""

    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(prog=str(self.__class__.__name__))
        self.classweighting = False
        self.parser.add_argument("--classweighting", action='store_true',
                                 help="Weight classes in loss computation.")
        self.classes = None

        self._parsed_args = None
        # get all available losses
        _dotmod = importlib.util.find_spec(Path(os.path.dirname(__file__)).name)
        for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
            importlib.import_module('.' + name, _dotmod.name)
        self._avail_losses = {str(cls.__name__).lower(): cls for cls in BaseLoss.__subclasses__()}

        with tf.name_scope("loss/") as scp:
            self.scope = scp

    def __call__(self, logits, indices):
        raise NotImplementedError

    def setup(self):
        pass

    def parse_known_args(self, *args, **kwargs):
        r = self.parser.parse_known_args(args[0], self)
        self._parsed_args = args[0]
        self.setup()
        return r


def register_parser(parser) -> dict:
    import importlib.util
    import pkgutil
    import os
    from pathlib import Path

    _dotmod = importlib.util.find_spec(Path(os.path.dirname(__file__)).name)
    for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
        importlib.import_module('.' + name, _dotmod.name)
    _classes = {str(cls.__name__).lower(): cls for cls in BaseLoss.__subclasses__()}

    avail_methods = []
    for _cls in _classes:
        # add argument to group
        avail_methods.append(str(_cls).lower())

    parser.add_argument('-l', '--loss', choices=avail_methods, dest="loss", required=True)
    parser.add_argument('--loss-monitor', choices=avail_methods, dest="loss_monitor", nargs='+')

    return _classes
