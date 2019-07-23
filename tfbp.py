#!/usr/bin/env python3


if __name__ == '__main__':

    try:
        import argparse
        import importlib

        parser = argparse.ArgumentParser(
            description='TensorFlow Boilerplate (tfbp).',
            prog="tfbp")

        datasampler = importlib.import_module("data")
        data_options = datasampler.register_parser(parser)

        experiments = importlib.import_module("experiment")
        exp_options = experiments.register_parser(parser)

        losses = importlib.import_module("loss")
        loss_options = losses.register_parser(parser)

        models = importlib.import_module("model")
        model_options = models.register_parser(parser)

        # parse args known to tfbp
        args, unknown_args = parser.parse_known_args()
        # choose classes from possible options
        sampler = data_options[args.data]
        experiment = exp_options[args.experiment]
        loss = loss_options[args.loss]
        model = model_options[args.model]

        # instantiate and parse args known to each subclass
        sampler = sampler()
        if unknown_args:
            _ = sampler.parse_known_args(unknown_args)
        else:
            _ = sampler.parse_known_args("".split())
        loss = loss()
        if unknown_args:
            _ = loss.parse_known_args(unknown_args)
        else:
            _ = loss.parse_known_args("".split())
        model = model()
        if unknown_args:
            _ = model.parse_known_args(unknown_args)
        else:
            _ = model.parse_known_args("".split())

        # compose complete experiment, parse args known to experiment
        experiment = experiment(sampler, model, loss)
        if unknown_args:
            _ = experiment.parse_known_args(unknown_args)
        else:
            _ = experiment.parse_known_args("".split())

        experiment.execute(args.option, args.loss_monitor)

    # catch KeyboardInterrupt error message
    # IT WAS INTENTIONAL
    except KeyboardInterrupt:
        pass
