import numpy as np
import os
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__.split('.')[-1])

try:
    from equations import *
    from atmospheres import *
except:
    from sys import path
    path.insert(0, './stratified_dynamics')
    from stratified_dynamics.equations import *
    from stratified_dynamics.atmospheres import *

class FC_polytrope_2d(FC_equations_2d, Polytrope):
    def __init__(self, dimensions=2, *args, **kwargs):
        super(FC_polytrope_2d, self).__init__(dimensions=dimensions) 
        Polytrope.__init__(self, dimensions=dimensions, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_polytrope_2d, self).set_equations(*args, **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)
    
    def initialize_output(self, solver, data_dir, *args, **kwargs):
        super(FC_polytrope_2d, self).initialize_output(solver, data_dir, *args, **kwargs)

        #This creates an output file that contains all of the useful atmospheric info at the beginning of the run
        import h5py
        import os
        from dedalus.core.field import Field
        dir = data_dir + '/atmosphere/'
        file = dir + 'atmosphere.h5'
        if self.domain.dist.comm_cart.rank == 0:
            if not os.path.exists('{:s}'.format(dir)):
                os.mkdir('{:s}'.format(dir))
        if self.domain.dist.comm_cart.rank == 0:
            f = h5py.File('{:s}'.format(file), 'w')
        key_set = list(self.problem.parameters.keys())
        extended_keys = ['chi','nu','del_chi','del_nu']
        key_set.extend(extended_keys)
        logger.debug("Outputing atmosphere parameters for {}".format(key_set))
        for key in key_set:
            if 'scale' in key:
                continue
            if key in extended_keys:
                field_key = True
            elif type(self.problem.parameters[key]) == Field:
                field_key = True
                self.problem.parameters[key].set_scales(1, keep_data=True)
            else:
                field_key = False
            if field_key:
                try:
                    array = self.problem.parameters[key]['g'][0,:]
                except:
                    if key == 'chi':
                        array = self.problem.parameters['chi_l']['g'][0,:] +\
                                self.problem.parameters['chi_r']['g'][0,:]
                    elif key == 'nu':
                        array = self.problem.parameters['nu_l']['g'][0,:] +\
                                self.problem.parameters['nu_r']['g'][0,:]
                    elif key == 'del_chi':
                        array = self.problem.parameters['del_chi_l']['g'][0,:] +\
                                self.problem.parameters['del_chi_r']['g'][0,:]
                    elif key == 'del_nu':
                        array = self.problem.parameters['del_nu_l']['g'][0,:] +\
                                self.problem.parameters['del_nu_r']['g'][0,:]
                    else:
                        logger.error("key error on atmosphere output {}".format(key))
                        
                this_chunk      = np.zeros(self.nz)
                global_chunk    = np.zeros(self.nz)
                n_per_cpu       = int(self.nz/self.domain.dist.comm_cart.size)
                i_chunk_0 = self.domain.dist.comm_cart.rank*(n_per_cpu)
                i_chunk_1 = (self.domain.dist.comm_cart.rank+1)*(n_per_cpu)
                this_chunk[i_chunk_0:i_chunk_1] = array
                self.domain.dist.comm_cart.Allreduce(this_chunk, global_chunk, op=MPI.SUM)
                if self.domain.dist.comm_cart.rank == 0:
                    f[key] = global_chunk                        
            elif self.domain.dist.comm_cart.rank == 0:
                f[key] = self.problem.parameters[key]
                
        if self.domain.dist.comm_cart.rank == 0:
            f['dimensions']     = 2
            f['nx']             = self.nx
            f['nz']             = self.nz
            f['z']              = self.domain.grid(axis=-1, scales=1)
            f['m_ad']           = self.m_ad
            f['m']              = self.m_ad - self.epsilon
            f['epsilon']        = self.epsilon
            f['n_rho_cz']       = self.n_rho_cz
            f['rayleigh']       = self.Rayleigh
            f['prandtl']        = self.Prandtl
            f['aspect_ratio']   = self.aspect_ratio
            f['atmosphere_name']= self.atmosphere_name
            f.close()
            
        return self.analysis_tasks

class FC_polytrope_2d_kappa(FC_equations_2d_kappa, Polytrope):
    def __init__(self, dimensions=2, *args, **kwargs):
        super(FC_polytrope_2d_kappa, self).__init__(dimensions=dimensions) 
        Polytrope.__init__(self, dimensions=dimensions, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def initialize_output(self, solver, data_dir, *args, **kwargs):
        super(FC_polytrope_2d_kappa, self).initialize_output(solver, data_dir, *args, **kwargs)

        #This creates an output file that contains all of the useful atmospheric info at the beginning of the run
        import h5py
        import os
        from dedalus.core.field import Field
        dir = data_dir + '/atmosphere/'
        file = dir + 'atmosphere.h5'
        if self.domain.dist.comm_cart.rank == 0:
            if not os.path.exists('{:s}'.format(dir)):
                os.mkdir('{:s}'.format(dir))
        if self.domain.dist.comm_cart.rank == 0:
            f = h5py.File('{:s}'.format(file), 'w')
        key_set = list(self.problem.parameters.keys())
        logger.debug("Outputing atmosphere parameters for {}".format(key_set))
        for key in key_set:
            if 'scale' in key:
                continue
            if type(self.problem.parameters[key]) == Field:
                field_key = True
                self.problem.parameters[key].set_scales(1, keep_data=True)
            else:
                field_key = False
            if field_key:
                try:
                    array = self.problem.parameters[key]['g'][0,:]
                except:
                    logger.error("key error on atmosphere output {}".format(key))
                        
                this_chunk      = np.zeros(self.nz)
                global_chunk    = np.zeros(self.nz)
                n_per_cpu       = int(self.nz/self.domain.dist.comm_cart.size)
                i_chunk_0 = self.domain.dist.comm_cart.rank*(n_per_cpu)
                i_chunk_1 = (self.domain.dist.comm_cart.rank+1)*(n_per_cpu)
                this_chunk[i_chunk_0:i_chunk_1] = array
                self.domain.dist.comm_cart.Allreduce(this_chunk, global_chunk, op=MPI.SUM)
                if self.domain.dist.comm_cart.rank == 0:
                    f[key] = global_chunk                        
            elif self.domain.dist.comm_cart.rank == 0:
                f[key] = self.problem.parameters[key]
                
        if self.domain.dist.comm_cart.rank == 0:
            f['dimensions']     = 2
            f['nx']             = self.nx
            f['nz']             = self.nz
            f['z']              = self.domain.grid(axis=-1, scales=1)
            f['m_ad']           = self.m_ad
            f['m']              = self.m_ad - self.epsilon
            f['epsilon']        = self.epsilon
            f['n_rho_cz']       = self.n_rho_cz
            f['rayleigh']       = self.Rayleigh
            f['prandtl']        = self.Prandtl
            f['aspect_ratio']   = self.aspect_ratio
            f['atmosphere_name']= self.atmosphere_name
            f.close()
            
        return self.analysis_tasks
                     
class FC_polytrope_3d(FC_equations_3d, Polytrope):
    def __init__(self, dimensions=3, *args, **kwargs):
        super(FC_polytrope_3d, self).__init__(dimensions=dimensions) 
        Polytrope.__init__(self, dimensions=dimensions, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        super(FC_polytrope_3d, self).set_equations(*args, **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)

