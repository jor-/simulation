import util.logging
import ndop.model.eval

with util.logging.Logger():
    m = ndop.model.eval.Model()
    m._parameter_db.merge_file_db_to_array_db()
