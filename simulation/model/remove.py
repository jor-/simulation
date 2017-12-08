import simulation
import simulation.model.options
import simulation.model.eval


def remove(model_name, concentrations_index, parameter_set_index=None, time_step=1, use_constant_concentrations=True):

    # prepare model and options
    model_options = simulation.model.options.ModelOptions()
    model_options.model_name = model_name
    model_options.time_step = time_step

    m = simulation.model.eval.Model(model_options=model_options)

    # get concentration_db
    if use_constant_concentrations:
        concentration_db = m._constant_concentrations_db
    else:
        concentration_db = m._vector_concentrations_db

    # remove concentration index if no parameter index specified
    if parameter_set_index is None:
        concentration_db.remove_index(concentrations_index, force=True)
    # else remove parameter index
    else:
        # get parameter_db
        model_options.initial_concentration_options.concentrations = concentration_db.get_value(concentrations_index)
        parameter_db = m._parameters_db

        # remove parameter index
        parameter_db.remove_index(parameter_set_index, force=True)
        if parameter_db.number_of_indices() == 0:
            concentration_db.remove_index(concentrations_index, force=True)


# *** main function for script call *** #

def _main():

    # parse arguments
    import argparse
    import simulation.model.constants
    import util.logging

    parser = argparse.ArgumentParser(description='Removing values from the database.')

    parser.add_argument('--model_name', default=simulation.model.constants.MODEL_NAMES[0], choices=simulation.model.constants.MODEL_NAMES, help='The name of the model that should be used.')
    parser.add_argument('--time_step', type=int, default=1, help='The time step of the model that should be used. Default: 1')
    parser.add_argument('--use_vector_concentrations', action='store_true', help='Remove one entry for vector concentrations and not for constant concentrations.')
    parser.add_argument('--concentrations_index', type=int, required=True, help='The concentration index that should be used.')
    parser.add_argument('--parameter_set_index', type=int, default=None, help='The model parameter index that should be used. If none is specified, all parameter sets for the concentration index are removed.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))

    args = parser.parse_args()

    # call function
    with util.logging.Logger():
        remove(args.model_name, args.concentrations_index, args.parameter_set_index, time_step=args.time_step, use_constant_concentrations=not args.use_vector_concentrations)


if __name__ == "__main__":
    _main()
