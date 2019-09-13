def _main():
    import argparse

    import util.logging
    import simulation.model.constants
    import simulation.plot.model

    with util.logging.Logger():
        parser = argparse.ArgumentParser(description='Plotting model output.')

        parser.add_argument('model_name', choices=simulation.model.constants.MODEL_NAMES, help='The name of the model that should be used.')
        parser.add_argument('--concentrations_index', type=int, help='The constant concentration index that should be used if no constant concentration values are specified.')
        parser.add_argument('--parameter_set_index', type=int, help='The model parameter index that should be used if no model parameters are specified.')
        parser.add_argument('--tracer', type=str, default=None)
        parser.add_argument('--time_dim', type=int, default=1)
        parser.add_argument('--path', type=str, default='/tmp/')
        parser.add_argument('--y_max', type=float, default=None)

        simulation.plot.model.output(args.model_name, concentrations_index=args.concentrations_index, parameters_index=args.parameters_index, tracer=args.tracer, time_dim=args.time_dim, path=args.path, y_max=args.y_max)


if __name__ == "__main__":
    _main()
