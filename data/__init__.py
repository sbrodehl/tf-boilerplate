import argparse
import importlib.util
import pkgutil
import os
from pathlib import Path


class BaseDataSampler(object):
    """DataSampler which generates a TensorFlow Dataset object from given input."""

    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(prog=str(self.__class__.__name__))
        # get all available losses
        _dotmod = importlib.util.find_spec(Path(os.path.dirname(__file__)).name)
        for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
            importlib.import_module('.' + name, _dotmod.name)
        self._avail_samplers = {str(cls.__name__).lower(): cls for cls in BaseDataSampler.__subclasses__()}

        self.classes = None

    def training(self):
        raise NotImplementedError

    def testing(self):
        raise NotImplementedError

    def validation(self):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def get_output_types(self) -> tuple:
        raise NotImplementedError

    def get_output_shapes(self) -> tuple:
        raise NotImplementedError

    def visualize(self, data, *args, **kwargs) -> bool:
        raise NotImplementedError

    def setup(self):
        pass

    def parse_known_args(self, *args, **kwargs):
        r = self.parser.parse_known_args(args[0], self)
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
    _classes = {str(cls.__name__).lower(): cls for cls in BaseDataSampler.__subclasses__()}

    avail_methods = []
    for _cls in _classes:
        # add argument to group
        avail_methods.append(str(_cls).lower())

    parser.add_argument('-d', '--data', choices=avail_methods, dest="data", required=True)

    return _classes
