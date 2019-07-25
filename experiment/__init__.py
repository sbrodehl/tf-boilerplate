import argparse
from pathlib import Path
import datetime
import importlib.util
import pkgutil
import os
import multiprocessing
from data import BaseDataSampler
from loss import BaseLoss
from model import BaseModel


_EXP_OPTIONS = dict()


def register_experiment_run(func):
    """Register a function as a plug-in"""
    _EXP_OPTIONS[func.__name__] = func
    return func


class BaseExperiment(object):
    """Experiment which does things."""

    def __init__(self, sampler: BaseDataSampler, model: BaseModel, lossfn: BaseLoss):
        self.parser = argparse.ArgumentParser(prog=str(self.__class__.__name__))
        self.epoch = 5
        self.parser.add_argument("--epoch", type=int, default=self.epoch,
                                 help="training epochs")
        self.batchsize = 16
        self.parser.add_argument("--batchsize", type=int, default=self.batchsize,
                                 help="batch size")
        self.logdir = "/data/projects/DWD/log"
        self.parser.add_argument("--logdir", type=str, default=self.logdir,
                                 help="log folder")
        self.savemins = 5
        self.parser.add_argument("--savemins", type=int, default=self.savemins,
                                 help="""Save the graph and summaries of once every N steps.""")
        self.logsteps = 1
        self.parser.add_argument("--logsteps", type=int, default=self.logsteps,
                                 help="""Log the values of once every N steps.""")
        self.sampler = sampler
        self.lossfn = lossfn
        self.model = model

        self.procs = int(multiprocessing.cpu_count()*0.66)

        # get all available experiments
        _dotmod = importlib.util.find_spec(Path(os.path.dirname(__file__)).name)
        for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
            importlib.import_module('.' + name, _dotmod.name)
        self._avail_experiments = {str(cls.__name__).lower(): cls for cls in BaseExperiment.__subclasses__()}

        # addition private variables
        self.prevrun = False
        self.loss_monitoring = []

    @register_experiment_run
    def training(self):
        pass

    @register_experiment_run
    def testing(self):
        pass

    @register_experiment_run
    def validation(self):
        pass

    @register_experiment_run
    def debug(self):
        pass

    def execute(self, option, loss_monitoring=None):
        # monitor various losses during execution
        if loss_monitoring is None:
            loss_monitoring = []
        self.loss_monitoring = loss_monitoring
        func = getattr(self, option, None)
        if func is None:
            raise RuntimeError("Selected option ({}) not found.".format(option))
        return func()

    def setup(self):
        # check if log dir is empty / exists
        logdir = self.logdir
        plogdir = Path(self.logdir)
        dttm = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if plogdir.exists() and plogdir.is_dir():
            if not list(plogdir.glob("*.pbtxt")) \
                   + list(plogdir.glob("*.ckpt*")) \
                   + list(plogdir.glob("*tfevents*")):
                logdir = str(Path(self.logdir) / "{} {} {} {} {}".format(
                    dttm,
                    self.__class__.__name__, self.sampler.__class__.__name__, self.model.__class__.__name__, self.lossfn.__class__.__name__))
            else:
                self.prevrun = True
                print("WARNING: Found previous tf run in log dir!")
        else:
            logdir = str(Path(self.logdir) / "{} {} {} {} {}".format(
                    dttm,
                    self.__class__.__name__, self.sampler.__class__.__name__, self.model.__class__.__name__, self.lossfn.__class__.__name__))
        self.logdir = logdir

        # create dir if its not exists
        Path(self.logdir).mkdir(parents=True)

        # dump command line to log directory
        with open(str(Path(self.logdir) / "cmd.log"), "w") as f:
            import sys
            f.write(" ".join(sys.argv))

        # spread love like I'm paddington bear
        self.model.classes = self.sampler.classes
        self.lossfn.classes = self.sampler.classes

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
    _classes = {str(cls.__name__).lower(): cls for cls in BaseExperiment.__subclasses__()}

    avail_methods = []
    for _cls in _classes:
        # add argument to group
        avail_methods.append(str(_cls).lower())

    parser.add_argument('-e', '--experiment', choices=avail_methods, dest="experiment", required=True)
    parser.add_argument('-o', '--option', choices=list(_EXP_OPTIONS.keys()), dest="option", required=True)

    return _classes
