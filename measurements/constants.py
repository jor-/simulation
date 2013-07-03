import os.path

from ndop.constants import BASE_DIR

BASE_DIR = os.path.join(BASE_DIR, 'measurement_data')

WOA_PO4_NOBS_NETCDF_MONTHLY_FILE =  os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_monthly_nobs.cdf')
WOA_PO4_NOBS_NETCDF_ANNUAL_FILE =  os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_annual_nobs.cdf')
WOA_PO4_NOBS_NETCDF_DATANAME = 'PO4NOBS'

WOA_PO4_VARIS_NETCDF_MONTHLY_FILE = os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_monthly_vari.cdf')
WOA_PO4_VARIS_NETCDF_ANNUAL_FILE = os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_annual_vari.cdf')
WOA_PO4_VARIS_NETCDF_DATANAME = 'PO4VARI'

WOA_PO4_MEANS_NETCDF_MONTHLY_FILE = os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_monthly_mean.cdf')
WOA_PO4_MEANS_NETCDF_ANNUAL_FILE = os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_annual_mean.cdf')
WOA_PO4_MEANS_NETCDF_DATANAME = 'PO4MEAN'

WOA_PO4_MOS_NETCDF_MONTHLY_FILE =   os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_monthly_mos.cdf')
WOA_PO4_MOS_NETCDF_ANNUAL_FILE =   os.path.join(BASE_DIR, 'PO4/woa09/woa09_po4_2.8x2.8_annual_mos.cdf')
WOA_PO4_MOS_NETCDF_DATANAME = 'PO4MOS'

PO4_NOBS =     os.path.join(BASE_DIR, 'PO4/po4_2.8x2.8_monthly_nobs.npy')
PO4_VARIS =    os.path.join(BASE_DIR, 'PO4/po4_2.8x2.8_monthly_vari.npy')
PO4_MEANS =    os.path.join(BASE_DIR, 'PO4/po4_2.8x2.8_monthly_mean.npy')
PO4_MOS =  os.path.join(BASE_DIR, 'PO4/po4_2.8x2.8_monthly_mos.npy')

PO4_ANNUAL_THRESHOLD = 500




YOSHIMURA_DOP_MEASUREMENT_FILE = os.path.join(BASE_DIR, 'DOP/yoshimura2007/data_values.txt')

DOP_NOBS =    os.path.join(BASE_DIR, 'DOP/dop_2.8x2.8_monthly_nobs.npy')
DOP_VARIS =   os.path.join(BASE_DIR, 'DOP/dop_2.8x2.8_monthly_vari.npy')
DOP_MEANS =   os.path.join(BASE_DIR, 'DOP/dop_2.8x2.8_monthly_mean.npy')
DOP_MOS = os.path.join(BASE_DIR, 'DOP/dop_2.8x2.8_monthly_mos.npy')



NOBS =    os.path.join(BASE_DIR, 'dop_po4_2.8x2.8_monthly_nobs.npy')
VARIS =   os.path.join(BASE_DIR, 'dop_po4_2.8x2.8_monthly_vari.npy')
MEANS =   os.path.join(BASE_DIR, 'dop_po4_2.8x2.8_monthly_mean.npy')
MOS = os.path.join(BASE_DIR, 'dop_po4_2.8x2.8_monthly_mos.npy')