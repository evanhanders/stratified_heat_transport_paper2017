import numpy as np
import scipy.special as scp
import os
from mpi4py import MPI

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from tools import analysis
except:
    from sys import path
    path.insert(0, './tools')
    from ..tools import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from dedalus import public as de

class Atmosphere:
    def __init__(self, verbose=False, fig_dir='./', dimensions=2, **kwargs):
        self._set_domain(**kwargs)
        
        self.make_plots = verbose
        self.fig_dir = fig_dir + '/'
        self.dimensions = dimensions
        
        if self.fig_dir[-1] != '/':
            self.fig_dir += '/'
        if self.domain.dist.comm_cart.rank == 0 and not os.path.exists(self.fig_dir):
            os.mkdir(self.fig_dir)
                
    def _set_domain(self, nx=256, Lx=4,
                          ny=256, Ly=4,
                          nz=128, Lz=1,
                          grid_dtype=np.float64, comm=MPI.COMM_WORLD, mesh=None):
        self.mesh=mesh

        z_basis = de.Chebyshev('z', nz, interval=[0., Lz], dealias=3/2)
        if self.dimensions > 1:
            x_basis = de.Fourier(  'x', nx, interval=[0., Lx], dealias=3/2)
        if self.dimensions > 2:
            y_basis = de.Fourier(  'y', ny, interval=[0., Ly], dealias=3/2)
        if self.dimensions == 1:
            bases = [z_basis]
        elif self.dimensions == 2:
            bases = [x_basis, z_basis]
        elif self.dimensions == 3:
            bases = [x_basis, y_basis, z_basis]
        else:
            logger.error('>3 dimensions not implemented')
        
        self.domain = de.Domain(bases, grid_dtype=grid_dtype, comm=comm, mesh=mesh)
        
        self.z = self.domain.grid(-1) # need to access globally-sized z-basis
        self.Lz = self.domain.bases[-1].interval[1] - self.domain.bases[-1].interval[0] # global size of Lz
        self.nz = self.domain.bases[-1].coeff_size

        self.z_dealias = self.domain.grid(axis=-1, scales=self.domain.dealias)

        if self.dimensions == 1:
            self.x, self.Lx, self.nx, self.delta_x = None, 0, None, None
            self.y, self.Ly, self.ny, self.delta_y = None, 0, None, None
        if self.dimensions > 1:
            self.x = self.domain.grid(0)
            self.Lx = self.domain.bases[0].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.nx = self.domain.bases[0].coeff_size
            self.delta_x = self.Lx/self.nx
        if self.dimensions > 2:
            self.y = self.domain.grid(1)
            self.Ly = self.domain.bases[1].interval[1] - self.domain.bases[0].interval[0] # global size of Lx
            self.ny = self.domain.bases[1].coeff_size
            self.delta_y = self.Ly/self.ny
        

    def filter_field(self, field,frac=0.25, fancy_filter=False):
        dom = field.domain
        logger.info("filtering field with frac={}".format(frac))
        if fancy_filter:
            logger.debug("filtering using field_filter approach.  Please check.")
            local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
            coeff = []
            for i in range(dom.dim)[::-1]:
                logger.info("i = {}".format(i))
                coeff.append(np.linspace(0,1,dom.global_coeff_shape[i],endpoint=False))
            logger.info(coeff)
            cc = np.meshgrid(*coeff)
            field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')

            for i in range(len(cc)):
                logger.info("cc {} shape {}".format(i, cc[i].shape))
                logger.info("local slice {}".format(local_slice[i]))
                logger.info("field_filter shape {}".format(field_filter.shape))

        
            for i in range(dom.dim):
                logger.info("trying i={}".format(i))
                field_filter = field_filter | (cc[i][local_slice[i]] > frac)
        
            # broken for 3-D right now; works for 2-D.  Nope, broken now in 2-D as well... what did I do?
            field['c'][field_filter] = 0j
        else:
            logger.debug("filtering using set_scales approach.  Please check.")
            orig_scale = field.meta[:]['scale']
            field.set_scales(frac, keep_data=True)
            field['c']
            field['g']
            field.set_scales(orig_scale, keep_data=True)
            
    def _new_ncc(self):
        field = self.domain.new_field()
        if self.dimensions > 1:
            field.meta['x']['constant'] = True
        if self.dimensions > 2:
            field.meta['y']['constant'] = True            
        return field

    def _new_field(self):
        field = self.domain.new_field()
        return field

    def get_problem(self):
        return self.problem

    def evaluate_at_point(self, f, z=0):
        return f.interpolate(z=z)

    def value_at_boundary(self, field):
        orig_scale = field.meta[:]['scale']
        try:
            field_top    = self.evaluate_at_point(field, z=self.Lz)['g'][0][0]
            if not np.isfinite(field_top):
                logger.info("Likely interpolation error at top boundary; setting field=1")
                logger.info("orig_scale: {}".format(orig_scale))
                field_top = 1
            field_bottom = self.evaluate_at_point(field, z=0)['g'][0][0]
            field.set_scales(orig_scale, keep_data=True)
        except:
            logger.debug("field at top shape {}".format(field['g'].shape))
            field_top = None
            field_bottom = None
        
        return field_bottom, field_top
    
    def _set_atmosphere(self):
        self.necessary_quantities = OrderedDict()

        self.phi = self._new_ncc()
        self.necessary_quantities['phi'] = self.phi

        self.del_ln_rho0 = self._new_ncc()
        self.rho0 = self._new_ncc()
        self.necessary_quantities['del_ln_rho0'] = self.del_ln_rho0
        self.necessary_quantities['rho0'] = self.rho0

        self.del_s0 = self._new_ncc()
        self.necessary_quantities['del_s0'] = self.del_s0
        
        self.T0_zz = self._new_ncc()
        self.T0_z = self._new_ncc()
        self.T0 = self._new_ncc()
        self.necessary_quantities['T0_zz'] = self.T0_zz
        self.necessary_quantities['T0_z'] = self.T0_z
        self.necessary_quantities['T0'] = self.T0

        self.del_P0 = self._new_ncc()
        self.P0 = self._new_ncc()
        self.necessary_quantities['del_P0'] = self.del_P0
        self.necessary_quantities['P0'] = self.P0

        self.nu = self._new_ncc()
        self.chi = self._new_ncc()
        self.nu_l = self._new_ncc()
        self.chi_l = self._new_ncc()
        self.del_chi_l = self._new_ncc()
        self.del_nu_l = self._new_ncc()
        self.necessary_quantities['nu_l'] = self.nu_l
        self.necessary_quantities['chi_l'] = self.chi_l
        self.necessary_quantities['del_chi_l'] = self.del_chi_l
        self.necessary_quantities['del_nu_l'] = self.del_nu_l
        self.nu_r = self._new_ncc()
        self.chi_r = self._new_ncc()
        self.del_chi_r = self._new_ncc()
        self.del_nu_r = self._new_ncc()
        self.necessary_quantities['nu_r'] = self.nu_r
        self.necessary_quantities['chi_r'] = self.chi_r
        self.necessary_quantities['del_chi_r'] = self.del_chi_r
        self.necessary_quantities['del_nu_r'] = self.del_nu_r

        self.scale = self._new_ncc()
        self.scale_continuity = self._new_ncc()
        self.scale_energy = self._new_ncc()
        self.scale_momentum = self._new_ncc()
        self.necessary_quantities['scale'] = self.scale
        self.necessary_quantities['scale_continuity'] = self.scale_continuity
        self.necessary_quantities['scale_energy'] = self.scale_energy
        self.necessary_quantities['scale_momentum'] = self.scale_momentum


    def _set_parameters(self):
        '''
        Basic parameters needed for any stratified atmosphere.
        '''
        self.problem.parameters['Lz'] = self.Lz
        if self.dimensions > 1:
            self.problem.parameters['Lx'] = self.Lx
        if self.dimensions > 2:
            self.problem.parameters['Ly'] = self.Ly

        self.problem.parameters['gamma'] = self.gamma
        self.problem.parameters['Cv'] = 1/(self.gamma-1)
        self.problem.parameters['Cv_inv'] = self.gamma-1
        self.problem.parameters['Cp'] = self.gamma/(self.gamma-1)
        self.problem.parameters['Cp_inv'] = (self.gamma-1)/self.gamma

        # the following quantities must be calculated and are missing
        # from the atmosphere stub.

        # thermodynamic quantities
        self.problem.parameters['T0'] = self.T0
        self.problem.parameters['T0_z'] = self.T0_z
        self.problem.parameters['T0_zz'] = self.T0_zz
        
        self.problem.parameters['rho0'] = self.rho0
        self.problem.parameters['del_ln_rho0'] = self.del_ln_rho0
                    
        self.problem.parameters['del_s0'] = self.del_s0

        # gravity
        self.problem.parameters['g']  = self.g
        self.problem.parameters['phi']  = self.phi

        # scaling factor to reduce NCC bandwidth of all equations
        self.problem.parameters['scale'] = self.scale
        self.problem.parameters['scale_continuity'] = self.scale_continuity
        self.problem.parameters['scale_momentum'] = self.scale_momentum
        self.problem.parameters['scale_energy'] = self.scale_energy

        # diffusivities
        self.problem.parameters['nu_l'] = self.nu_l
        self.problem.parameters['chi_l'] = self.chi_l
        self.problem.parameters['del_chi_l'] = self.del_chi_l
        self.problem.parameters['del_nu_l'] = self.del_nu_l
        self.problem.parameters['nu_r'] = self.nu_r
        self.problem.parameters['chi_r'] = self.chi_r
        self.problem.parameters['del_chi_r'] = self.del_chi_r
        self.problem.parameters['del_nu_r'] = self.del_nu_r

        # Cooling
        self.problem.parameters['Qcool_z'] = 0

    def copy_atmosphere(self, atmosphere):
        '''
        Copies values from a target atmosphere into the current atmosphere.
        '''
        self.necessary_quantities = atmosphere.necessary_quantities
            
    def plot_atmosphere(self):

        for key in self.necessary_quantities:
            logger.debug("plotting atmospheric quantity {}".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(2,1,1)
            quantity = self.necessary_quantities[key]
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], quantity['g'][0,:])
            if np.min(quantity['g'][0,:]) != np.max(quantity['g'][0,:]):
                ax.set_ylim(np.min(quantity['g'][0,:])-0.05*np.abs(np.min(quantity['g'][0,:])),
                        np.max(quantity['g'][0,:])+0.05*np.abs(np.max(quantity['g'][0,:])))
            ax.set_xlabel('z')
            ax.set_ylabel(key)
            
            ax = fig_q.add_subplot(2,1,2)
            power_spectrum = np.abs(quantity['c'][0,:]*np.conj(quantity['c'][0,:]))
            ax.plot(np.arange(len(quantity['c'][0,:])), power_spectrum)
            ax.axhline(y=1e-20, color='black', linestyle='dashed') # ncc_cutoff = 1e-10
            ax.set_xlabel('z')
            ax.set_ylabel("Tn power spectrum: {}".format(key))
            ax.set_yscale("log", nonposy='clip')
            ax.set_xscale("log", nonposx='clip')

            fig_q.savefig("atmosphere_{}_p{}.png".format(key, self.domain.distributor.rank), dpi=300)
            plt.close(fig_q)

        for key in self.necessary_quantities:
            if key not in ['P0', 'rho0']:
                continue
            logger.debug("plotting atmosphereic quantity ln({})".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(1,1,1)
            quantity = self.necessary_quantities[key]
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], np.log(quantity['g'][0,:]))
            if np.min(quantity['g'][0,:]) != np.max(quantity['g'][0,:]):
                ax.set_ylim(np.min(np.log(quantity['g'][0,:]))-0.05*np.abs(np.min(np.log(quantity['g'][0,:]))),
                        np.max(np.log(quantity['g'][0,:]))+0.05*np.abs(np.max(np.log(quantity['g'][0,:]))))
            ax.set_xlabel('z')
            ax.set_ylabel('ln_'+key)
            fig_q.savefig(self.fig_dir+"atmosphere_ln_{}_p{}.png".format(key, self.domain.distributor.rank), dpi=300, bbox_inches='tight')
            plt.close(fig_q)
      
        fig_atm = plt.figure()
        axT = fig_atm.add_subplot(2,2,1)
        axT.plot(self.z[0,:], self.T0['g'][0,:])
        axT.set_ylabel('T0')
        axP = fig_atm.add_subplot(2,2,2)
        axP.semilogy(self.z[0,:], self.P0['g'][0,:]) 
        axP.set_ylabel('P0')
        axR = fig_atm.add_subplot(2,2,3)
        axR.semilogy(self.z[0,:], self.rho0['g'][0,:])
        axR.set_ylabel(r'$\rho0$')
        axS = fig_atm.add_subplot(2,2,4)
        analysis.semilogy_posneg(axS, self.z[0,:], self.del_s0['g'][0,:], color_neg='red')
        
        axS.set_ylabel(r'$\nabla s0$')
        fig_atm.savefig("atmosphere_quantities_p{}.png".format(self.domain.distributor.rank), dpi=300)

        fig_atm = plt.figure()
        axS = fig_atm.add_subplot(2,2,1)
        axdelS = fig_atm.add_subplot(2,2,2)
        axlnP = fig_atm.add_subplot(2,2,3)
        axdellnP = fig_atm.add_subplot(2,2,4)

        Cv_inv = self.gamma-1
        axS.plot(self.z[0,:], 1/Cv_inv*np.log(self.T0['g'][0,:]) - 1/Cv_inv*(self.gamma-1)*np.log(self.rho0['g'][0,:]), label='s0', linewidth=2)
        axS.plot(self.z[0,:], (1+(self.gamma-1)/self.gamma*self.g)*np.log(self.T0['g'][0,:]), label='s based on lnT', linewidth=2)
        axS.plot(self.z[0,:], np.log(self.T0['g'][0,:]) - (self.gamma-1)/self.gamma*np.log(self.P0['g'][0,:]), label='s based on lnT and lnP', linewidth=2)
        
        axdelS.plot(self.z[0,:], self.del_s0['g'][0,:], label=r'$\nabla s0$', linewidth=2)
        axdelS.plot(self.z[0,:], self.T0_z['g'][0,:]/self.T0['g'][0,:] + self.g*(self.gamma-1)/self.gamma*1/self.T0['g'][0,:],
                    label=r'$\nabla s0$ from T0', linewidth=2, linestyle='dashed',color='red')
         
        axlnP.plot(self.z[0,:], np.log(self.P0['g'][0,:]), label='ln(P)', linewidth=2)
        axlnP.plot(self.z[0,:], self.ln_P0['g'][0,:], label='lnP', linestyle='dashed', linewidth=2)
        axlnP.plot(self.z[0,:], -self.g*np.log(self.T0['g'][0,:])*(self.T0_z['g'][0,:]), label='-g*lnT', linewidth=2, linestyle='dotted')
        
        axdellnP.plot(self.z[0,:], self.del_ln_P0['g'][0,:], label='dellnP', linewidth=2)
        axdellnP.plot(self.z[0,:], -self.g/self.T0['g'][0,:], label='-g/T', linestyle='dashed', linewidth=2, color='red')
        
        #axS.legend()
        axS.set_ylabel(r'$s0$')
        fig_atm.savefig("atmosphere_s0_p{}.png".format(self.domain.distributor.rank), dpi=300)

    def plot_scaled_atmosphere(self):

        for key in self.necessary_quantities:
            logger.debug("plotting atmospheric quantity {}".format(key))
            fig_q = plt.figure()
            ax = fig_q.add_subplot(2,1,1)
            quantity = self.necessary_quantities[key]
            quantity['g'] *= self.scale['g']
            quantity.set_scales(1, keep_data=True)
            ax.plot(self.z[0,:], quantity['g'][0,:])
            ax.set_xlabel('z')
            ax.set_ylabel(key+'*scale')

            ax = fig_q.add_subplot(2,1,2)
            ax.plot(np.arange(len(quantity['c'][0,:])), np.abs(quantity['c'][0,:]*np.conj(quantity['c'][0,:])))
            ax.set_xlabel('z')
            ax.set_ylabel("Tn power spectrum: {}*scale".format(key))
            ax.set_yscale("log", nonposy='clip')
            ax.set_xscale("log", nonposx='clip')
            
            fig_q.savefig("atmosphere_{}scale_p{}.png".format(key, self.domain.distributor.rank), dpi=300)
            plt.close(fig_q)
                        
    def check_that_atmosphere_is_set(self):
        for key in self.necessary_quantities:
            quantity = self.necessary_quantities[key]['g']
            quantity_set = quantity.any()
            if not quantity_set:
                logger.info("WARNING: atmosphere {} is all zeros on process 0".format(key))
                
    def test_hydrostatic_balance(self, P_z=None, P=None, T=None, rho=None, make_plots=False):

        if rho is None:
            logger.error("HS balance test requires rho (currently)")
            raise
        
        if P_z is None:
            if P is None:
                if T is None:
                    logger.error("HS balance test requires P_z, P or T")
                    raise
                else:
                    T_scales = T.meta[:]['scale']
                    rho_scales = rho.meta[:]['scale']
                    if rho_scales != 1:
                        rho.set_scales(1, keep_data=True)
                    if T_scales != 1:
                        T.set_scales(1, keep_data=True)
                    P = self._new_field()
                    T.set_scales(self.domain.dealias, keep_data=True)
                    rho.set_scales(self.domain.dealias, keep_data=True)
                    P.set_scales(self.domain.dealias, keep_data=False)
                    P['g'] = T['g']*rho['g']
                    T.set_scales(T_scales, keep_data=True)
                    rho.set_scales(rho_scales, keep_data=True)

            P_z = self._new_field()
            P.differentiate('z', out=P_z)
            P_z.set_scales(1, keep_data=True)

        rho_scales = rho.meta[:]['scale']
        rho.set_scales(1, keep_data=True)
        # error in hydrostatic balance diagnostic
        HS_balance = P_z['g']+self.g*rho['g']
        relative_error = HS_balance/P_z['g']
        rho.set_scales(rho_scales, keep_data=True)
        
        HS_average = self._new_field()
        HS_average['g'] = HS_balance
        if self.dimensions > 1:
            HS_average.integrate('x')
            HS_average['g'] /= self.Lx
        HS_average.set_scales(1, keep_data=True)

        relative_error_avg = self._new_field()
        relative_error_avg['g'] = relative_error
        if self.dimensions > 1:
            relative_error_avg.integrate('x')
            relative_error_avg['g'] /= self.Lx
        relative_error_avg.set_scales(1, keep_data=True)

        if self.make_plots or make_plots:
            fig = plt.figure()
            ax1 = fig.add_subplot(2,1,1)
            if self.dimensions > 1:
                ax1.plot(self.z[0,:], P_z['g'][0,:])
                ax1.plot(self.z[0,:], -self.g*rho['g'][0,:])
            else:
                ax1.plot(self.z[:], P_z['g'][:])
                ax1.plot(self.z[:], -self.g*rho['g'][:])
            ax1.set_ylabel(r'$\nabla P$ and $\rho g$')
            ax1.set_xlabel('z')

            ax2 = fig.add_subplot(2,1,2)
            if self.dimensions > 1:
                ax2.semilogy(self.z[0,:], np.abs(relative_error[0,:]))
                ax2.semilogy(self.z[0,:], np.abs(relative_error_avg['g'][0,:]))
            else:
                ax2.semilogy(self.z[:], np.abs(relative_error[:]))
                ax2.semilogy(self.z[:], np.abs(relative_error_avg['g'][:]))
            ax2.set_ylabel(r'$|\nabla P + \rho g |/|\nabla P|$')
            ax2.set_xlabel('z')
            fig.savefig(self.fig_dir+"atmosphere_HS_balance_p{}.png".format(self.domain.distributor.rank), dpi=300)

        max_rel_err = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error)), op=MPI.MAX)
        max_rel_err_avg = self.domain.dist.comm_cart.allreduce(np.max(np.abs(relative_error_avg['g'])), op=MPI.MAX)
        logger.info('max error in HS balance: point={} avg={}'.format(max_rel_err, max_rel_err_avg))

    def check_atmosphere(self, make_plots=False, **kwargs):
        if self.make_plots or make_plots:
            try:
                self.plot_atmosphere()
            except:
                logger.info("Problems in plot_atmosphere: atm full of NaNs?")
        self.test_hydrostatic_balance(make_plots=make_plots, **kwargs)
        self.check_that_atmosphere_is_set()

class Polytrope(Atmosphere):
    '''
    Single polytrope, stable or unstable.
    '''
    def __init__(self,
                 nx=256, Lx=None,
                 ny=256, Ly=None,
                 nz=128, Lz=None,
                 aspect_ratio=4,
                 n_rho_cz = 3,
                 m_cz=None, epsilon=1e-4, gamma=5/3,
                 constant_kappa=True, constant_mu=True,
                 **kwargs):
        
        self.atmosphere_name = 'single polytrope'
        self.aspect_ratio    = aspect_ratio
        self.n_rho_cz        = n_rho_cz

        self._set_atmosphere_parameters(gamma=gamma, epsilon=epsilon, poly_m=m_cz)
        if m_cz is None:
            m_cz = self.poly_m

        if Lz is None:
            if n_rho_cz is not None:
                Lz = self._calculate_Lz_cz(n_rho_cz, m_cz)
            else:
                logger.error("Either Lz or n_rho must be set")
                raise
        if Lx is None:
            Lx = Lz*aspect_ratio
        if Ly is None:
            Ly = Lx
            
        super(Polytrope, self).__init__(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, **kwargs)
        logger.info("   Lx = {:g}, Lz = {:g}".format(self.Lx, self.Lz))
        self.z0 = 1. + self.Lz
       
        self.constant_kappa = constant_kappa
        self.constant_mu    = constant_mu
        if self.constant_kappa == False and self.constant_mu == False:
            self.constant_diffusivities = True
        else:
            self.constant_diffusivities = False

        self._set_atmosphere()
        self._set_timescales()

    def _calculate_Lz_cz(self, n_rho_cz, m_cz):
        '''
        Calculate Lz based on the number of density scale heights and the initial polytrope.
        '''
        #The absolute value allows for negative m_cz.
        Lz_cz = np.exp(n_rho_cz/np.abs(m_cz))-1
        return Lz_cz
    
    def _set_atmosphere_parameters(self, gamma=5/3, epsilon=0, poly_m=None, g=None):
        # polytropic atmosphere characteristics
        self.gamma = gamma
        self.Cv = 1/(self.gamma-1)
        self.Cp = self.gamma*self.Cv
        self.epsilon = epsilon

        self.m_ad = 1/(self.gamma-1)

        # trap on poly_m/epsilon conflicts?
        if poly_m is None:
            self.poly_m = self.m_ad - self.epsilon
        else:
            self.poly_m = poly_m

        self.m_cz = self.poly_m

        if g is None:
            self.g = self.poly_m + 1
        else:
            self.g = g

        logger.info("polytropic atmosphere parameters:")
        logger.info("   poly_m = {:g}, epsilon = {:g}, gamma = {:g}".format(self.poly_m, self.epsilon, self.gamma))
    
    def _set_atmosphere(self):
        super(Polytrope, self)._set_atmosphere()

        self.del_ln_rho_factor = -self.poly_m
        self.del_ln_rho0['g'] = self.del_ln_rho_factor/(self.z0 - self.z)
        self.rho0['g'] = (self.z0 - self.z)**self.poly_m

        self.del_s0_factor = - self.epsilon 
        self.delta_s = self.del_s0_factor*np.log(self.z0)
        self.del_s0['g'] = self.del_s0_factor/(self.z0 - self.z)
 
        self.T0_zz['g'] = 0        
        self.T0_z['g'] = -1
        self.T0['g'] = self.z0 - self.z       

        self.P0['g'] = (self.z0 - self.z)**(self.poly_m+1)
        self.P0.differentiate('z', out=self.del_P0)
        self.del_P0.set_scales(1, keep_data=True)
        self.P0.set_scales(1, keep_data=True)
        
        if self.constant_diffusivities:
            self.scale['g']            = (self.z0 - self.z)
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g']   = (self.z0 - self.z)
            self.scale_energy['g']     = (self.z0 - self.z)
        else:
            # consider whether to scale nccs involving chi differently (e.g., energy equation)
            self.scale['g']            = (self.z0 - self.z)
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g']   = (self.z0 - self.z)# **np.ceil(self.m_cz)
            self.scale_energy['g']     = (self.z0 - self.z)# **np.ceil(self.m_cz)

        # choose a particular gauge for phi (g*z0); and -grad(phi)=g_vec=-g*z_hat
        # double negative is correct.
        self.phi['g'] = -self.g*(self.z0 - self.z)

        rho0_max, rho0_min = self.value_at_boundary(self.rho0)
        if rho0_max is not None:
            rho0_ratio = rho0_max/rho0_min
            logger.info("   density: min {}  max {}".format(rho0_min, rho0_max))
            logger.info("   density scale heights = {:g} (measured)".format(np.log(rho0_ratio)))
            logger.info("   density scale heights = {:g} (target)".format(np.log((self.z0)**self.poly_m)))
            
        H_rho_top = (self.z0-self.Lz)/self.poly_m
        H_rho_bottom = (self.z0)/self.poly_m
        logger.info("   H_rho = {:g} (top)  {:g} (bottom)".format(H_rho_top,H_rho_bottom))
        if self.delta_x != None:
            logger.info("   H_rho/delta x = {:g} (top)  {:g} (bottom)".format(H_rho_top/self.delta_x,
                                                                          H_rho_bottom/self.delta_x))
        
    def _set_timescales(self, atmosphere=None):
        if atmosphere is None:
            atmosphere=self
            
        # min of global quantity
        atmosphere.min_BV_time = self.domain.dist.comm_cart.allreduce(np.min(np.sqrt(np.abs(self.g*self.del_s0['g']/self.Cp))), op=MPI.MIN)
        atmosphere.freefall_time = np.sqrt(self.Lz/self.g)
        atmosphere.buoyancy_time = np.sqrt(self.Lz/self.g/np.abs(self.epsilon))
        
        logger.info("atmospheric timescales:")
        logger.info("   min_BV_time = {:g}, freefall_time = {:g}, buoyancy_time = {:g}".format(atmosphere.min_BV_time,
                                                                                               atmosphere.freefall_time,
                                                                                               atmosphere.buoyancy_time))
    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1, split_diffusivities=False):
        
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl

        # set nu and chi at top based on Rayleigh number
        self.nu_top = nu_top = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s/self.Cp)*self.g)/Rayleigh)
        self.chi_top = chi_top = nu_top/Prandtl

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = nu_top
            chi = chi_top

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))
        else:
            if self.constant_kappa:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    chi_l = chi_top/(self.rho0['g'])
                    chi_r = 0
                else:
                    if self.poly_m < 1:
                        chi_l = np.exp(self.n_rho_cz)*chi_top/(self.z0 - self.z)
                    else:
                        chi_l = chi_top/(self.z0 - self.z)
                    chi_r = chi_top/(self.rho0['g']) - chi_l
                logger.info('using constant kappa')
            else:
                chi_l = chi_top
                chi_r = 0
                logger.info('using constant chi')
            if self.constant_mu:
                self.rho0.set_scales(1, keep_data=True)
                if not split_diffusivities:
                    nu_l  = nu_top/(self.rho0['g'])
                    nu_r = 0
                else:
                    if self.poly_m < 1:
                        nu_l  = np.exp(self.n_rho_cz)*nu_top/(self.z0 - self.z)
                    else:
                        nu_l  = nu_top/(self.z0 - self.z)
                    nu_r  = nu_top/(self.rho0['g']) - nu_l
                logger.info('using constant mu')
            else:
                nu_l  = nu_top
                nu_r = 0
                logger.info('using constant nu')

      
            logger.info("   nu_top = {:g}, chi_top = {:g}".format(nu_top, chi_top))
                    
        #Allows for atmosphere reuse
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.set_scales(1, keep_data=True)
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
        self.nu_l['g'] = nu_l
        self.chi_l['g'] = chi_l
        self.nu_r['g'] = nu_r
        self.chi_r['g'] = chi_r
        self.nu['g'] = nu_l + nu_r
        self.chi['g'] = chi_l + chi_r

        self.chi_l.differentiate('z', out=self.del_chi_l)
        self.chi_l.set_scales(1, keep_data=True)
        self.nu_l.differentiate('z', out=self.del_nu_l)
        self.nu_l.set_scales(1, keep_data=True)
        self.chi_r.differentiate('z', out=self.del_chi_r)
        self.chi_r.set_scales(1, keep_data=True)
        self.nu_r.differentiate('z', out=self.del_nu_r)
        self.nu_r.set_scales(1, keep_data=True)

        # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
        self.thermal_time = self.Lz**2/(self.chi.interpolate(z=self.Lz/2)['g'][0])
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/(self.nu.interpolate(z=self.Lz/2)['g'][0])
        self.top_viscous_time = 1/nu_top

        if self.dimensions > 1:
            self.thermal_time = self.thermal_time[0]
            self.viscous_time = self.viscous_time[0]
        if self.dimensions > 2:
            self.thermal_time = self.thermal_time[0]
            self.viscous_time = self.viscous_time[0]

        logger.info("thermal_time = {}, top_thermal_time = {}".format(self.thermal_time,
                                                                      self.top_thermal_time))
        self.nu.set_scales(1, keep_data=True)
        self.chi.set_scales(1, keep_data=True)
