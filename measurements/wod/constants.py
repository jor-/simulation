import datetime

DATA_DIR = '/work_j2/sunip229/NDOP/measurement_data/PO4/wod/data'
CRUISES_PICKLED_FILE = '/work_j2/sunip229/NDOP/measurement_data/PO4/wod/analysis/cruises_list.py'
MEASUREMENTS_PICKLED_FILE = '/work_j2/sunip229/NDOP/measurement_data/PO4/wod/analysis/measurements_dict.py'

BASE_DATE = datetime.datetime(1770, 1, 1)
# SAME_DATETIME_BOUND = 1 / (366.0 * 2)

## netcdf datanames
DAY_OFFSET = 'time' # number of days since 01.01.1770 (float)
LAT = 'lat'
LON = 'lon'
DEPTH = 'z'
DEPTH_FLAG = 'z_WODflag'
PO4 = 'Phosphate'
PO4_FLAG = 'Phosphate_WODflag'
PO4_PROFILE_FLAG = 'Phosphate_WODprofileflag'
MISSING_VALUE = - 10**10

# 
# ## CRAP
# 
# DATE = 'time' # number of days since 01.01.1770
# TIME = 'GMT_time'