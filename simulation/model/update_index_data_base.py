import util.logging
import simulation.model.eval

with util.logging.Logger():
    m = simulation.model.eval.Model()
    m._parameter_db.merge_file_db_to_array_db()
