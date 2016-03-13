
import os
import time
import subprocess
import re
import numpy as np

import util.batch.universal.system
import util.io.fs

import util.logging
logger = util.logging.logger



class Metos3D_Job(util.batch.universal.system.Job):


    @property
    def last_spinup_line(self):
        self.wait_until_finished()

        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        last_spinup_line = None
        with open(self.output_file) as f:
            for line in f.readlines():
                if 'Spinup Function norm' in line:
                    last_spinup_line = line

        return last_spinup_line


    @property
    def last_year(self):
        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        last_spinup_line = self.last_spinup_line

        if last_spinup_line is not None:
            last_spinup_line = last_spinup_line.strip()
            last_spinup_year_str = last_spinup_line.split()[1]
            last_spinup_year = int(last_spinup_year_str) + 1
        else:
            last_spinup_year = 0

        return last_spinup_year


    @property
    def last_tolerance(self):
        # 9.704s 0010 Spinup Function norm 2.919666257647e+00
        last_spinup_line = self.last_spinup_line

        if last_spinup_line is not None:
            last_spinup_line = last_spinup_line.strip()
            last_spinup_tolerance_str = last_spinup_line.split()[5]
            last_spinup_tolerance = float(last_spinup_tolerance_str)
        else:
            last_spinup_tolerance = float('inf')

        return last_spinup_tolerance


    @property
    def time_step(self):
        from ndop.model.constants import METOS_T_DIM

        opt = self.options
        metos3d_opt_file = opt['/metos3d/option_file']
        with open(metos3d_opt_file) as metos3d_opt_file_object:
            metos3d_opt_lines = metos3d_opt_file_object.readlines()

        for metos3d_opt_line in metos3d_opt_lines:
            if re.search('Metos3DTimeStepCount', metos3d_opt_line) is not None:
                time_step_count = int(re.findall('\d+', metos3d_opt_line)[1])
                time_step = int(METOS_T_DIM / time_step_count)

        return time_step


    @property
    def tracer_input_path(self):
        opt = self.options

        try:
            input_dir = opt['/metos3d/tracer_input_path']
            input_filename = opt['/metos3d/po4_input_filename']
        except KeyError:
            input_filename = None

        if input_filename is not None:
            input_file = os.path.join(input_dir, input_filename)
            real_input_file = os.path.realpath(input_file)
            tracer_input_path = os.path.dirname(real_input_file)
        else:
            tracer_input_path = None

        return tracer_input_path



    @property
    def metos3d_option_file(self):
        return self.options['/metos3d/option_file']

    @property
    def model_parameters_file(self):
        return self.options['/model/parameters_file']

    @property
    def dop_output_file(self):
        try:
            output_path = self.options['/metos3d/tracer_output_path']
        except KeyError:
            output_path = self.options['/metos3d/output_path']
        return os.path.join(output_path, self.options['/metos3d/dop_output_filename'])

    @property
    def dop_output_info_file(self):
        return self.dop_output_file + '.info'

    @property
    def po4_output_file(self):
        try:
            output_path = self.options['/metos3d/tracer_output_path']
        except KeyError:
            output_path = self.options['/metos3d/output_path']
        return os.path.join(output_path, self.options['/metos3d/po4_output_filename'])

    @property
    def po4_output_info_file(self):
        return self.po4_output_file + '.info'
    

    @property
    def exit_code(self):
        ## get metos3d exit code
        exit_code = super().exit_code
        if exit_code != 0:
            return exit_code
        
        ## check if output file exists
        if self.output_file is not None and not os.path.exists(self.output_file):
            ValueError('Output file {} does not exist. The job is not finished'.format(self.output_file))

        ## check output file for errors
        IGNORE_ERRORS = ('Error_Path = ', 'cpuinfo: error while loading shared libraries: libgcc_s.so.1: cannot open shared object file: No such file or directory')
        output = self.output
        for ingore_error in IGNORE_ERRORS:
            output = output.replace(ingore_error, '')
        output = output.lower()
        if 'error' in output:
            return 255
        else:
            return 0
    

    def is_finished(self, check_exit_code=True):
        ## check if finished without exit code check
        if not super().is_finished(check_exit_code=False):
            return False
        
        ## ensure that output file exists for error check
        if check_exit_code and self.output_file is not None and not os.path.exists(self.output_file):
            return False

        ## check if finished with exit code check
        if check_exit_code and not super().is_finished(check_exit_code=check_exit_code):
            return False
            
        ## check if output file is completely written
        job_output = self.output
        if 'Metos3DFinal' in job_output:
            return True
        else:
            time.sleep(30)
            if self.output != job_output:
                return False
            else:
                raise util.batch.universal.system.JobError(self.id, self.output_dir, 'The job output file is not completely written!', job_output)
    
    

    def make_read_only_input(self, read_only=True):
        super().make_read_only_input(read_only=read_only)
        if read_only:
            util.io.fs.make_read_only(self.metos3d_option_file)
            util.io.fs.make_read_only(self.model_parameters_file)

    def make_read_only_output(self, read_only=True):
        super().make_read_only_output(read_only=read_only)
        if read_only:
            util.io.fs.make_read_only(self.dop_output_file)
            util.io.fs.make_read_only(self.dop_output_info_file)
            util.io.fs.make_read_only(self.po4_output_file)
            util.io.fs.make_read_only(self.po4_output_info_file)




    def update_output_dir(self, new_output_path):
        opt = self.options
        old_output_path = opt['/metos3d/output_path']

        if old_output_path.endswith('/'):
            old_output_path = old_output_path[:-1]
        if new_output_path.endswith('/'):
            new_output_path = new_output_path[:-1]

        opt.replace_all_str_options(old_output_path, new_output_path)


    @staticmethod
    def best_nodes_setup(years, node_kind=None, nodes_max=None):
        from ndop.model.constants import JOB_MEMORY_GB, JOB_MIN_CPUS

        logger.debug('Getting best nodes_setup for {} years with node_kind {} and nodes_max {}.'.format(years, node_kind, nodes_max))

        ## min cpus
        cpus_min = min(int(np.ceil(years/10)), JOB_MIN_CPUS)

        ## max nodes
        if years <= 1:
            if nodes_max is not None:
                # nodes_max = util.io.io.get_sequence_from_values_or_file(nodes_max)
                nodes_max = list(nodes_max)
                for i in range(len(nodes_max)):
                    nodes_max[i] = min(nodes_max[i], 1)
            else:
                nodes_max = 1

        ## best node setup
        nodes_setup = util.batch.universal.system.NodeSetup(memory=JOB_MEMORY_GB, node_kind=node_kind, total_cpus_min=cpus_min, nodes_max=nodes_max)
        logger.debug('Best nodes_setup is {}.'.format(nodes_setup))

        return nodes_setup



    def write_job_file(self, model_parameters, years, tolerance, time_step=1, write_trajectory=False, tracer_input_path=None, job_setup=None):
        from ndop.model.constants import JOB_OPTIONS_FILENAME, JOB_MEMORY_GB, MODEL_PARAMETERS_FORMAT_STRING, MODEL_PARAMETERS_FORMAT_STRING_OLD_STYLE, METOS_T_DIM, METOS_DATA_DIR, METOS_SIM_FILE

        logger.debug('Initialising job with job_setup {}.'.format(job_setup))


        ## check input
        if METOS_T_DIM % time_step != 0:
            raise ValueError('Wrong time_step in model options. {} has to be divisible by time_step. But time_step is {}.'.format(METOS_T_DIM, time_step))

        ## unpack job setup
        if job_setup is not None:
            try:
                job_name = job_setup['name']
            except KeyError:
                job_name = 'Metos3D'
            try:
                nodes_setup = job_setup['nodes_setup']
            except KeyError:
                nodes_setup = None
            try:
                nodes_max = job_setup['nodes_max']
            except KeyError:
                nodes_max = None
        else:
            job_name = ''
            nodes_setup = None
            nodes_max = None

        ## prepare job name
        if len(job_name) > 0:
            job_name += '_'
        job_name += '{}_{}'.format(years, time_step)


        ## use best node setup if no node setup passed
        if nodes_setup is None:
            nodes_setup = self.best_nodes_setup(years, node_kind=node_kind, nodes_max=nodes_max)

        ## check/set walltime
        sec_per_year = np.exp(- (nodes_setup.nodes * nodes_setup.cpus) / 40) * 30 + 1.25
        sec_per_year /= time_step
        estimated_walltime_hours = np.ceil(years * sec_per_year / 60**2) + 1
        if nodes_setup.walltime is None:
            nodes_setup.walltime = estimated_walltime_hours
        else:
            if nodes_setup.walltime < estimated_walltime_hours:
                logger.debug('The chosen walltime {} for the job with {} years, {} nodes and {} cpus is below the estimated walltime {}.'.format(nodes_setup.walltime, years, nodes_setup.nodes, nodes_setup.cpus, estimated_walltime_hours))

        ## init job
        super().init_job_file(job_name, nodes_setup)


        ## get output dir
        output_dir = self.output_dir
        output_dir_not_expanded = os.path.join(self.output_dir_not_expanded, "") # ending with separator


        ## set model options
        opt = self.options

        model_parameters = np.array(model_parameters, dtype=np.float64)
        opt['/model/parameters'] = model_parameters
        opt['/model/parameters_file'] = os.path.join(output_dir_not_expanded, 'model_parameter.txt')
        np.savetxt(opt['/model/parameters_file'], opt['/model/parameters'], fmt=MODEL_PARAMETERS_FORMAT_STRING_OLD_STYLE)

        time_step_count = int(METOS_T_DIM / time_step)
        opt['/model/time_step_count'] = time_step_count
        opt['/model/time_step'] = 1 / time_step_count


        ## set metos3d options
        opt['/metos3d/data_path'] = METOS_DATA_DIR
        opt['/metos3d/sim_file'] = METOS_SIM_FILE
        opt['/metos3d/years'] = years
        opt['/metos3d/write_trajectory'] = write_trajectory
        if tolerance is not None:
            opt['/metos3d/tolerance'] = tolerance

        if write_trajectory:
            tracer_output_dir = os.path.join(output_dir, 'trajectory/')
            os.makedirs(tracer_output_dir, exist_ok=True)
            tracer_output_dir_not_expanded = os.path.join(output_dir_not_expanded, 'trajectory/')
            opt['/metos3d/tracer_output_path'] = tracer_output_dir_not_expanded

        opt['/metos3d/output_path'] = output_dir_not_expanded
        opt['/metos3d/option_file'] = os.path.join(output_dir_not_expanded, 'metos3d_options.txt')
        opt['/metos3d/debuglevel'] = 1
        opt['/metos3d/po4_output_filename'] = 'po4_output.petsc'
        opt['/metos3d/dop_output_filename'] = 'dop_output.petsc'

        if tracer_input_path is not None:
            opt['/metos3d/po4_input_filename'] = 'po4_input.petsc'
            opt['/metos3d/dop_input_filename'] = 'dop_input.petsc'

            tracer_input_dir = os.path.relpath(tracer_input_path, start=output_dir)

            os.symlink(os.path.join(tracer_input_dir, opt['metos3d/po4_output_filename']), os.path.join(output_dir, opt['/metos3d/po4_input_filename']))
            os.symlink(os.path.join(tracer_input_dir, opt['metos3d/dop_output_filename']), os.path.join(output_dir, opt['/metos3d/dop_input_filename']))

            opt['/metos3d/tracer_input_path'] = output_dir_not_expanded

        model_parameters_string = str(tuple(map(lambda f: MODEL_PARAMETERS_FORMAT_STRING.format(f), model_parameters)))
        model_parameters_string = model_parameters_string.replace("'", '').replace('(', '').replace(')', '').replace(' ','')

        opt['/metos3d/parameters_string'] = model_parameters_string


        ## write metos3d option file
        f = open(opt['/metos3d/option_file'], mode='w')

        f.write('# debug \n')
        f.write('-Metos3DDebugLevel                      %i \n\n' % opt['/metos3d/debuglevel'])

        f.write('# geometry \n')
        f.write('-Metos3DGeometryType                    Profile \n')
        f.write('-Metos3DProfileInputDirectory           %s/Geometry/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DProfileIndexStartFile           gStartIndices.bin \n')
        f.write('-Metos3DProfileIndexEndFile             gEndIndices.bin \n\n')

        f.write('# bgc tracer \n')
        f.write('-Metos3DTracerCount                     2 \n')

        try:
            f.write('-Metos3DTracerInputDirectory            %s \n' % opt['/metos3d/tracer_input_path'])
            f.write('-Metos3DTracerInitFile                  %s,%s \n' % (opt['/metos3d/po4_input_filename'], opt['/metos3d/dop_input_filename']))
        except KeyError:
            f.write('-Metos3DTracerInitValue                 2.17e+0,1.e-4 \n')

        try:
            f.write('-Metos3DTracerOutputDirectory           %s \n' % opt['/metos3d/tracer_output_path'])
        except KeyError:
            f.write('-Metos3DTracerOutputDirectory           %s \n' % opt['/metos3d/output_path'])

        f.write('-Metos3DTracerOutputFile                %s,%s \n\n' % (opt['/metos3d/po4_output_filename'], opt['/metos3d/dop_output_filename']))

        f.write('# bgc parameter \n')
        f.write('-Metos3DParameterCount                  7 \n')
        f.write('-Metos3DParameterValue                  %s \n\n' % opt['/metos3d/parameters_string'])

        f.write('# bgc boundary conditions \n')
        f.write('-Metos3DBoundaryConditionCount          2 \n')
        f.write('-Metos3DBoundaryConditionInputDirectory %s/Forcing/BoundaryCondition/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DBoundaryConditionName           Latitude,IceCover \n')
        f.write('-Metos3DLatitudeCount                   1 \n')
        f.write('-Metos3DLatitudeFileFormat              latitude.petsc \n')
        f.write('-Metos3DIceCoverCount                   12 \n')
        f.write('-Metos3DIceCoverFileFormat              fice_$02d.petsc \n\n')

        f.write('# bgc domain conditions \n')
        f.write('-Metos3DDomainConditionCount            2 \n')
        f.write('-Metos3DDomainConditionInputDirectory   %s/Forcing/DomainCondition/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DDomainConditionName             LayerDepth,LayerHeight \n')
        f.write('-Metos3DLayerDepthCount                 1 \n')
        f.write('-Metos3DLayerDepthFileFormat            z.petsc \n\n')
        f.write('-Metos3DLayerHeightCount                1 \n')
        f.write('-Metos3DLayerHeightFileFormat           dz.petsc \n')

        f.write('# transport \n')
        f.write('-Metos3DTransportType                   Matrix \n')
        f.write('-Metos3DMatrixInputDirectory            %s/Transport/Matrix5_4/1dt/ \n' % opt['/metos3d/data_path'])
        f.write('-Metos3DMatrixCount                     12 \n')
        f.write('-Metos3DMatrixExplicitFileFormat        Ae_$02d.petsc \n')
        f.write('-Metos3DMatrixImplicitFileFormat        Ai_$02d.petsc \n\n')

        f.write('# time stepping \n')
        f.write('-Metos3DTimeStepStart                   0.0 \n')
        f.write('-Metos3DTimeStepCount                   %i \n' % opt['/model/time_step_count'])
        f.write('-Metos3DTimeStep                        %.18f \n\n' % opt['/model/time_step'])

        f.write('# solver \n')
        f.write('-Metos3DSolverType                      Spinup \n')
        f.write('-Metos3DSpinupMonitor \n')
        try:
            f.write('-Metos3DSpinupTolerance                 %f \n' % opt['/metos3d/tolerance'])
        except KeyError:
            pass
        f.write('-Metos3DSpinupCount                     %i \n' % opt['/metos3d/years'])

        if opt['/metos3d/write_trajectory']:
            f.write('-Metos3DSpinupMonitorFileFormatPrefix   sp$0004d-,ts$0004d- \n')
            f.write('-Metos3DSpinupMonitorModuloStep         1,1 \n')

        util.io.fs.flush_and_close(f)


        ## write job file
        run_command = '{} {} \n'.format(opt['/metos3d/sim_file'], opt['/metos3d/option_file'])
        super().write_job_file(run_command, modules=['intel', 'intelmpi', 'petsc'])

        logger.debug('Job initialised.')
