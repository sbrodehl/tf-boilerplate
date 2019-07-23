import argparse
import importlib.util
import pkgutil
import os
from pathlib import Path


class BaseModel(object):
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(prog=str(self.__class__.__name__))

        # get all available models
        _dotmod = importlib.util.find_spec(Path(os.path.dirname(__file__)).name)
        for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
            importlib.import_module('.' + name, _dotmod.name)
        self._avail_models = {str(cls.__name__).lower(): cls for cls in BaseModel.__subclasses__()}

        self.classes = None

    def setup(self):
        pass

    def parse_known_args(self, *args, **kwargs):
        r = self.parser.parse_known_args(args[0], self)
        self.setup()
        return r

    def network(self, tensors):
        raise NotImplementedError


def register_parser(parser) -> dict:
    import importlib.util
    import pkgutil
    import os
    from pathlib import Path

    _dotmod = importlib.util.find_spec(Path(os.path.dirname(__file__)).name)
    for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
        importlib.import_module('.' + name, _dotmod.name)
    _classes = {str(cls.__name__).lower(): cls for cls in BaseModel.__subclasses__()}

    avail_methods = []
    for _cls in _classes:
        # add argument to group
        avail_methods.append(str(_cls).lower())

    parser.add_argument('-m', '--model', choices=avail_methods, dest="model", required=True)

    return _classes
