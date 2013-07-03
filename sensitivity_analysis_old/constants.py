woa_po4_nobs_file = '/work_j2/sunip229/NDOP/measurement_data/PO4/woa09/woa09_2.8x2.8_monthly_nobs_po4.cdf'
woa_po4_nobs_dataname = 'PO4NOBS'
woa_po4_vari_file = '/work_j2/sunip229/NDOP/measurement_data/PO4/woa09/woa09_2.8x2.8_monthly_vari_po4.cdf'
woa_po4_vari_dataname = 'PO4VARI'
woa_po4_mean_file = '/work_j2/sunip229/NDOP/measurement_data/PO4/woa09/woa09_2.8x2.8_monthly_mean_po4.cdf'
woa_po4_mean_dataname = 'PO4MEAN'


yoshimura2007_dop_measurement_file = '/work_j2/sunip229/NDOP/measurement_data/DOP/yoshimura2007/data_values.txt'


metos_land_sea_mask_petsc = '/work_j2/sunip229/NDOP/metos3d_data/landSeaMask.petsc'
metos_land_sea_mask_npy = '/work_j2/sunip229/NDOP/metos3d_data/landSeaMask.npy'

metos_trajectory_file_pattern = '/work_j2/sunip229/NDOP/metos3d_data/parameter_set_%p/trajectory_1_%n/trajectory/sp0000-ts%04d-%t_output.petsc'
metos_h_file_pattern = '/work_j2/sunip229/NDOP/metos3d_data/parameter_set_%p/trajectory_1_%n/model_h.txt'

metos_parameter_set_pattern = '%p'
metos_derivative_number_pattern = '%n'
metos_tracer_pattern = '%t'
metos_time_pattern = '%04d'
metos_time_pattern_length_str = 4

metos_tracers = ['dop', 'po4']
metos_t_length = 2880 
metos_derivative_length = 7
metos_z = [50, 120, 220, 360, 550, 790, 1080, 1420, 1810, 2250, 2740, 3280, 3870, 4510, 5200]


oed_F_file_pattern = '/work_j2/sunip229/NDOP/oed_data/parameter_set_%p/F_%t.npy'
oed_J_file_pattern = '/work_j2/sunip229/NDOP/oed_data/parameter_set_%p/J_%t.npy'
oed_dop_vari_file_pattern = '/work_j2/sunip229/NDOP/oed_data/parameter_set_%p/dop_vari_%t.npy'
oed_po4_vari_file_pattern = '/work_j2/sunip229/NDOP/oed_data/parameter_set_%p/po4_vari_%t.npy'
oed_po4_vari_from_vari_data_file_pattern = '/work_j2/sunip229/NDOP/oed_data/po4_vari_from_vari_data.npy'
oed_FIM_file_pattern = '/work_j2/sunip229/NDOP/oed_data/parameter_set_%p/FIM.npy'
oed_fix_FIM_tracer_file_pattern = '/work_j2/sunip229/NDOP/oed_data/parameter_set_%p/fix_FIM_%T.npy'

oed_parameter_set_pattern = '%p'
oed_time_length_pattern = '%t'
oed_tracer_pattern = '%T'