import simulation
import simulation.model.options
import simulation.model.eval

import util.io.fs


def remove(model_name, concentrations_index, parameter_set_index=None, time_step=1, use_constant_concentrations=True):

    # prepare model and options
    model_options = simulation.model.options.ModelOptions()
    model_options.model_name = model_name
    m = simulation.model.eval.Model(model_options=model_options)

    # get concentration_db
    if use_constant_concentrations:
        concentration_db = m._constant_concentrations_db
    else:
        concentration_db = m._vector_concentrations_db

    # remove concentration index if no parameter and time step index is specified
    if parameter_set_index is None and time_step is None:
        concentration_db.remove_index(concentrations_index, force=True)
    else:
        # set concentration
        model_options.initial_concentration_options.concentrations = concentration_db.get_value(concentrations_index)
        # set time step
        model_options.time_step = time_step
        # remove time step dir if no parameter index but time step is specified
        if parameter_set_index is None:
            time_step_dir = m.time_step_dir
            util.io.fs.remove_recursively(time_step_dir, force=True, not_exist_okay=True, exclude_dir=False)
        # else remove parameter index
        else:
            parameter_db = m._parameters_db
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
    parser.add_argument('--time_step', type=int, default=None, help='The time step of the model that should be used. Default: 1')
    parser.add_argument('--use_vector_concentrations', action='store_true', help='Remove one entry for vector concentrations and not for constant concentrations.')
    parser.add_argument('--concentrations_index', type=int, required=True, help='The concentration index that should be used.')
    parser.add_argument('--parameter_set_index', type=int, default=None, help='The model parameter index that should be used. If none is specified, all parameter sets for the concentration index are removed.')
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(simulation.__version__))

    args = parser.parse_args()

    # default values
    parameter_set_index = args.parameter_set_index
    time_step = args.time_step
    if time_step is None and parameter_set_index is not None:
        time_step = 1

    # call function
    with util.logging.Logger():
        remove(args.model_name, args.concentrations_index, parameter_set_index, time_step=time_step, use_constant_concentrations=not args.use_vector_concentrations)


if __name__ == "__main__":
    _main()
