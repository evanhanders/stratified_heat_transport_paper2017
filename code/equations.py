import numpy as np
import scipy.special as scp
import os
from mpi4py import MPI

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analysis

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from tools.EVP import EVP_homogeneous
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
        self.del_chi = self._new_ncc()
        self.del_nu = self._new_ncc()
        self.necessary_quantities['nu'] = self.nu
        self.necessary_quantities['chi'] = self.chi
        self.necessary_quantities['del_chi'] = self.del_chi
        self.necessary_quantities['del_nu'] = self.del_nu

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
        self.problem.parameters['nu'] = self.nu
        self.problem.parameters['chi'] = self.chi
        self.problem.parameters['del_chi'] = self.del_chi
        self.problem.parameters['del_nu'] = self.del_nu

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
            self.scale['g'] = self.z0 - self.z
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g'] = (self.z0 - self.z)
            self.scale_energy['g'] = (self.z0 - self.z)
        else:
            # consider whether to scale nccs involving chi differently (e.g., energy equation)
            self.scale['g'] = (self.z0 - self.z)
            self.scale_continuity['g'] = (self.z0 - self.z)
            self.scale_momentum['g'] = (self.z0 - self.z)
            self.scale_energy['g'] = (self.z0 - self.z)

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
    def _set_diffusivities(self, Rayleigh=1e6, Prandtl=1):
        
        logger.info("problem parameters:")
        logger.info("   Ra = {:g}, Pr = {:g}".format(Rayleigh, Prandtl))
        self.Rayleigh, self.Prandtl = Rayleigh, Prandtl

        # set nu and chi at top based on Rayleigh number
        nu_top = np.sqrt(Prandtl*(self.Lz**3*np.abs(self.delta_s/self.Cp)*self.g)/Rayleigh)
        chi_top = nu_top/Prandtl

        if self.constant_diffusivities:
            # take constant nu, chi
            nu = nu_top
            chi = chi_top

            logger.info("   using constant nu, chi")
            logger.info("   nu = {:g}, chi = {:g}".format(nu, chi))
        else:
            if self.constant_kappa:
                self.rho0.set_scales(1, keep_data=True)
                chi = chi_top/(self.rho0['g'])
                logger.info('using constant kappa')
            else:
                chi = chi_top
                logger.info('using constant chi')
            if self.constant_mu:
                self.rho0.set_scales(1, keep_data=True)
                nu  = nu_top/(self.rho0['g'])
                logger.info('using constant mu')
            else:
                nu  = nu_top
                logger.info('using constant nu')

      
            logger.info("   nu_top = {:g}, chi_top = {:g}".format(nu_top, chi_top))
                    
        #Allows for atmosphere reuse
        self.chi.set_scales(1, keep_data=True)
        self.nu.set_scales(1, keep_data=True)
        self.nu['g'] = nu
        self.chi['g'] = chi

        self.chi.differentiate('z', out=self.del_chi)
        self.chi.set_scales(1, keep_data=True)
        self.nu.differentiate('z', out=self.del_nu)
        self.nu.set_scales(1, keep_data=True)

        # determine characteristic timescales; use chi and nu at middle of domain for bulk timescales.
        self.thermal_time = self.Lz**2/self.chi.interpolate(z=self.Lz/2)['g'][0]
        self.top_thermal_time = 1/chi_top

        self.viscous_time = self.Lz**2/self.nu.interpolate(z=self.Lz/2)['g'][0]
        self.top_viscous_time = 1/nu_top

        if self.dimensions > 1:
            self.thermal_time = self.thermal_time[0]
            self.viscous_time = self.viscous_time[0]
        if self.dimensions > 2:
            self.thermal_time = self.thermal_time[0]
            self.viscous_time = self.viscous_time[0]

        logger.info("thermal_time = {}, top_thermal_time = {}".format(self.thermal_time,
                                                                          self.top_thermal_time))


class Equations():
    def __init__(self, dimensions=2):
        self.dimensions=dimensions
        pass
    
    def set_IVP_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        self.problem = de.IVP(self.domain, variables=self.variables, ncc_cutoff=ncc_cutoff)
        self.set_equations(*args, **kwargs)

    def set_eigenvalue_problem(self, *args, ncc_cutoff=1e-10, **kwargs):
        self.problem = EVP_homogeneous(self.domain, variables=self.variables, eigenvalue='omega', ncc_cutoff=ncc_cutoff)
        self.problem.substitutions['dt(f)'] = "omega*f"
        self.set_equations(*args, **kwargs)

    def at_boundary(self, f, z=0, tol=1e-12, derivative=False, dz=False, BC_text='BC'):        
        if derivative or dz:
            f_bc=self._new_ncc()
            f.differentiate('z', out=f_bc)
            BC_text += "_z"
        else:
            f_bc = f

        BC_field = f_bc.interpolate(z=z)

        try:
            BC = BC_field['g'][0][0]
        except:
            BC = BC_field['g']
            logger.error("shape of BC_field {}".format(BC_field['g'].shape))
            logger.error("BC = {}".format(BC))

        if np.abs(BC) < tol:
            BC = 0
        logger.info("Calculating boundary condition z={:7g}, {}={:g}".format(z, BC_text, BC))
        return BC

    def _set_subs(self):
        pass

    def global_noise(self, seed=42, **kwargs):            
        # Random perturbations, initialized globally for same results in parallel
        gshape = self.domain.dist.grid_layout.global_shape(scales=self.domain.dealias)
        slices = self.domain.dist.grid_layout.slices(scales=self.domain.dealias)
        rand = np.random.RandomState(seed=seed)
        noise = rand.standard_normal(gshape)[slices]

        # filter in k-space
        noise_field = self._new_field()
        noise_field.set_scales(self.domain.dealias, keep_data=False)
        noise_field['g'] = noise
        self.filter_field(noise_field, **kwargs)

        return noise_field

class FC_equations(Equations):
    def __init__(self, **kwargs):
        super(FC_equations, self).__init__(**kwargs)
        self.T1_left    = 0
        self.T1_right   = 0
        self.T1_z_left  = 0
        self.T1_z_right = 0

    def set_eigenvalue_problem_type_2(self, Rayleigh, Prandtl, **kwargs):
        self.problem = EVP_homogeneous(self.domain, variables=self.variables, eigenvalue='nu')
        self.problem.substitutions['dt(f)'] = "(0*f)"
        self.set_equations(Rayleigh, Prandtl, EVP_2 = True, **kwargs)

    def _set_subs(self):
        self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'

        # output parameters        
        self.problem.substitutions['rho_full'] = 'rho0*exp(ln_rho1)'
        self.problem.substitutions['rho_fluc'] = 'rho0*(exp(ln_rho1)-1)'
        self.problem.substitutions['ln_rho0']  = 'log(rho0)'
        self.problem.substitutions['T_full']            = '(T0 + T1)'
        self.problem.substitutions['ln_rho_full']       = '(ln_rho0 + ln_rho1)'

        self.problem.parameters['delta_s_atm'] = self.delta_s
        self.problem.substitutions['s_fluc'] = '(1/Cv_inv*log(1+T1/T0) - ln_rho1)'
        self.problem.substitutions['s_mean'] = '(1/Cv_inv*log(T0) - ln_rho0)'
        self.problem.substitutions['epsilon'] = 'plane_avg(log(T0**(1/(gamma-1))/rho0)/log(T0))'
        self.problem.substitutions['m_ad']    = '((gamma-1)**-1)'

        self.problem.substitutions['Rayleigh_global'] = 'g*Lz**3*delta_s_atm*Cp_inv/(nu*chi)'
        self.problem.substitutions['Rayleigh_local']  = 'g*Lz**4*dz(s_mean+s_fluc)*Cp_inv/(nu*chi)'
        
        self.problem.substitutions['vel_rms'] = 'sqrt(u**2 + v**2 + w**2)'
        self.problem.substitutions['KE'] = 'rho_full*(vel_rms**2)/2'
        self.problem.substitutions['PE'] = 'rho_full*phi'
        self.problem.substitutions['PE_fluc'] = 'rho_fluc*phi'
        self.problem.substitutions['IE'] = 'rho_full*Cv*(T1+T0)'
        self.problem.substitutions['IE_fluc'] = 'rho_full*Cv*T1+rho_fluc*Cv*T0'
        self.problem.substitutions['P'] = 'rho_full*(T1+T0)'
        self.problem.substitutions['P_fluc'] = 'rho_full*T1+rho_fluc*T0'
        self.problem.substitutions['h'] = 'IE + P'
        self.problem.substitutions['h_fluc'] = 'IE_fluc + P_fluc'
        self.problem.substitutions['u_rms'] = 'sqrt(u**2)'
        self.problem.substitutions['v_rms'] = 'sqrt(v**2)'
        self.problem.substitutions['w_rms'] = 'sqrt(w**2)'
        self.problem.substitutions['Re_rms'] = 'vel_rms*Lz/nu'
        self.problem.substitutions['Pe_rms'] = 'vel_rms*Lz/chi'
        self.problem.substitutions['Ma_iso_rms'] = '(vel_rms/sqrt(T_full))'
        self.problem.substitutions['Ma_ad_rms'] = '(vel_rms/(gamma*sqrt(T_full)))'
        #self.problem.substitutions['lambda_microscale'] = 'sqrt(plane_avg(vel_rms)/plane_avg(enstrophy))'
        #self.problem.substitutions['Re_microscale'] = 'vel_rms*lambda_microscale/nu'
        #self.problem.substitutions['Pe_microscale'] = 'vel_rms*lambda_microscale/chi'
        
        self.problem.substitutions['h_flux_z'] = 'w*h'
        self.problem.substitutions['kappa_flux_mean'] = '-rho0*chi*dz(T0)'
        self.problem.substitutions['kappa_flux_fluc'] = '-rho_full*chi*dz(T1) - rho_fluc*chi*dz(T0)'
        self.problem.substitutions['kappa_flux_z'] = '((kappa_flux_mean) + (kappa_flux_fluc))'
        self.problem.substitutions['KE_flux_z'] = 'w*KE'
        self.problem.substitutions['PE_flux_z'] = 'w*PE'
        #self.problem.substitutions['viscous_flux_z'] = '- rho_full * nu * (u*u_z + (4/3)*w*w_z + u*dx(w) - (2/3)*w*dx(u))'
        self.problem.substitutions['viscous_flux_z'] = '- rho_full * nu * (u*σxz + w*σzz)'
        self.problem.substitutions['convective_flux_z'] = '(viscous_flux_z + KE_flux_z + PE_flux_z + h_flux_z)'
        self.problem.substitutions['kappa_adiabatic_flux_z'] = '(rho0*chi*g/Cp)'
        self.problem.substitutions['kappa_reference_flux_z'] = '(-chi*rho0*(right(T1+T0)-left(T1+T0))/Lz)'
        self.problem.substitutions['Nusselt_norm'] = '(kappa_reference_flux_z-kappa_adiabatic_flux_z)'
        self.problem.substitutions['Nusselt'] = '((convective_flux_z+kappa_flux_z-kappa_adiabatic_flux_z)/(Nusselt_norm))'

    def set_BC(self,
               fixed_flux=None, fixed_temperature=None, mixed_flux_temperature=None, mixed_temperature_flux=None,
               stress_free=None, no_slip=None):

        if not(fixed_flux) and not(fixed_temperature) and not(mixed_temperature_flux) and not(mixed_flux_temperature):
            mixed_flux_temperature = True
        if not(stress_free) and not(no_slip):
            stress_free = True

        self.dirichlet_set = []

        # thermal boundary conditions
        # this needs to be done before any equations are entered
        #self.problem.parameters['T1_left']    = self.T1_left
        #self.problem.parameters['T1_right']   = self.T1_right  
        #self.problem.parameters['T1_z_left']  = self.T1_z_left 
        #self.problem.parameters['T1_z_right'] = self.T1_z_right 

        # thermal boundary conditions
        if fixed_flux:
            logger.info("Thermal BC: fixed flux (T1_z)")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc( "left(T1_z) = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
        elif fixed_temperature:
            logger.info("Thermal BC: fixed temperature (T1)")
            self.problem.add_bc( "left(T1) = 0")
            self.problem.add_bc("right(T1) = 0")
            self.dirichlet_set.append('T1')
        elif mixed_flux_temperature:
            logger.info("Thermal BC: mixed flux/temperature (T1_z/T1)")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc("left(T1_z) = 0")
            self.problem.add_bc("right(T1)  = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        elif mixed_temperature_flux:
            logger.info("Thermal BC: mixed temperature/flux (T1/T1_z)")
            logger.info("warning; these are not fully correct fixed flux conditions yet")
            self.problem.add_bc("left(T1)    = 0")
            self.problem.add_bc("right(T1_z) = 0")
            self.dirichlet_set.append('T1_z')
            self.dirichlet_set.append('T1')
        else:
            logger.error("Incorrect thermal boundary conditions specified")
            raise
        # horizontal velocity boundary conditions
        if stress_free:
            logger.info("Horizontal velocity BC: stress free")
            self.problem.add_bc( "left(u_z) = 0")
            self.problem.add_bc("right(u_z) = 0")
            self.dirichlet_set.append('u_z')
        elif no_slip:
            logger.info("Horizontal velocity BC: no slip")
            self.problem.add_bc( "left(u) = 0")
            self.problem.add_bc("right(u) = 0")
            self.dirichlet_set.append('u')
        else:
            logger.error("Incorrect horizontal velocity boundary conditions specified")
            raise

        # vertical velocity boundary conditions
        logger.info("Vertical velocity BC: impenetrable")
        self.problem.add_bc( "left(w) = 0")
        self.problem.add_bc("right(w) = 0")
        self.dirichlet_set.append('w')
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True

    def set_IC(self, solver, A0=1e-6, **kwargs):
        # initial conditions
        self.T_IC = solver.state['T1']
        self.ln_rho_IC = solver.state['ln_rho1']

        noise = self.global_noise(**kwargs)
        noise.set_scales(self.domain.dealias, keep_data=True)
        self.T_IC.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        z_dealias = self.domain.grid(axis=-1, scales=self.domain.dealias)
        self.T_IC['g'] = self.epsilon*A0*np.sin(np.pi*z_dealias/self.Lz)*noise['g']*self.T0['g']

        logger.info("Starting with T1 perturbations of amplitude A0 = {:g}".format(A0))

    def get_full_T(self, solver):
        T1 = solver.state['T1']
        T_scales = T1.meta[:]['scale']
        T1.set_scales(self.domain.dealias, keep_data=True)
        self.T0.set_scales(self.domain.dealias, keep_data=True)
        T = self._new_field()
        T.set_scales(self.domain.dealias, keep_data=False)
        T['g'] = self.T0['g'] + T1['g']
        T.set_scales(T_scales, keep_data=True)
        T1.set_scales(T_scales, keep_data=True)
        return T

    def get_full_rho(self, solver):
        ln_rho1 = solver.state['ln_rho1']
        rho_scales = ln_rho1.meta[:]['scale']
        rho = self._new_field()
        rho['g'] = self.rho0['g']*np.exp(ln_rho1['g'])
        rho.set_scales(rho_scales, keep_data=True)
        ln_rho1.set_scales(rho_scales, keep_data=True)
        return rho

    def check_system(self, solver, **kwargs):
        T = self.get_full_T(solver)
        rho = self.get_full_rho(solver)

        self.check_atmosphere(T=T, rho=rho, **kwargs)

    def set_eigenvalue_problem_type_2(self, Rayleigh, Prandtl, **kwargs):
        self.problem = EVP_homogeneous(self.domain, variables=self.variables, eigenvalue='nu')
        self.problem.substitutions['dt(f)'] = "(0*f)"
        self.set_equations(Rayleigh, Prandtl, EVP_2 = True, **kwargs)
        
    def initialize_output(self, solver, data_dir, full_output=False,
                          slices=[1,1], profiles=[1,1], scalar=[1,1], coeffs=[1,1], **kwargs):
        #  slices, profiles, and scalar are all [write_num, set_num]

        analysis_tasks = OrderedDict()
        self.analysis_tasks = analysis_tasks

        analysis_profile = solver.evaluator.add_file_handler(data_dir+"profiles", max_writes=20, parallel=False,
                                                             write_num=profiles[0], set_num=profiles[1],  **kwargs)
        analysis_profile.add_task("plane_avg(T1)", name="T1")
        analysis_profile.add_task("plane_avg(T_full)", name="T_full")
        analysis_profile.add_task("plane_avg(Ma_iso_rms)", name="Ma_iso")
        analysis_profile.add_task("plane_avg(Ma_ad_rms)", name="Ma_ad")
        analysis_profile.add_task("plane_avg(ln_rho1)", name="ln_rho1")
        analysis_profile.add_task("plane_avg(rho_full)", name="rho_full")
        analysis_profile.add_task("plane_avg(KE)", name="KE")
        analysis_profile.add_task("plane_avg(PE)", name="PE")
        analysis_profile.add_task("plane_avg(IE)", name="IE")
        analysis_profile.add_task("plane_avg(PE_fluc)", name="PE_fluc")
        analysis_profile.add_task("plane_avg(IE_fluc)", name="IE_fluc")
        analysis_profile.add_task("plane_avg(KE + PE + IE)", name="TE")
        analysis_profile.add_task("plane_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
        analysis_profile.add_task("plane_avg(w*(KE))", name="KE_flux_z")
        analysis_profile.add_task("plane_avg(w*(PE))", name="PE_flux_z")
        analysis_profile.add_task("plane_avg(w*(IE))", name="IE_flux_z")
        analysis_profile.add_task("plane_avg(w*(P))",  name="P_flux_z")
        analysis_profile.add_task("plane_avg(w*(h))",  name="enthalpy_flux_z")
        analysis_profile.add_task("plane_avg(viscous_flux_z)",  name="viscous_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z)", name="kappa_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_fluc)", name="kappa_flux_fluc_z")
        analysis_profile.add_task("plane_avg(kappa_flux_mean)", name="kappa_flux_mean_z")
        analysis_profile.add_task("plane_avg(w*(h))/plane_avg(Nusselt_norm)",  name="norm_enthalpy_flux_z")
        analysis_profile.add_task("plane_avg(viscous_flux_z)/plane_avg(Nusselt_norm)",  name="norm_viscous_flux_z")
        analysis_profile.add_task("plane_avg(w*(KE))/plane_avg(Nusselt_norm)", name="norm_KE_flux_z")
        analysis_profile.add_task("plane_avg(w*(PE))/plane_avg(Nusselt_norm)", name="norm_PE_flux_z")
        analysis_profile.add_task("plane_avg(kappa_flux_fluc)/plane_avg(Nusselt_norm)", name="norm_kappa_flux_fluc_z")
        analysis_profile.add_task("plane_avg(kappa_flux_z-kappa_adiabatic_flux_z)/plane_avg(Nusselt_norm)", name="norm_kappa_flux_z")
        analysis_profile.add_task("plane_avg(Nusselt)", name="Nusselt")
        analysis_profile.add_task("plane_avg(u_rms)", name="u_rms")
        analysis_profile.add_task("plane_avg(w_rms)", name="w_rms")
        analysis_profile.add_task("plane_avg(vel_rms)", name="vel_rms")
        analysis_profile.add_task("plane_avg(Re_rms)", name="Re_rms")
        analysis_profile.add_task("plane_avg(Pe_rms)", name="Pe_rms")
        analysis_profile.add_task("plane_avg(enstrophy)", name="enstrophy")
        analysis_profile.add_task("plane_std(enstrophy)", name="enstrophy_std")        
        analysis_profile.add_task("plane_avg(Rayleigh_global)", name="Rayleigh_global")
        analysis_profile.add_task("plane_avg(Rayleigh_local)",  name="Rayleigh_local")
        analysis_profile.add_task("plane_avg(s_fluc)", name="s_fluc")
        analysis_profile.add_task("plane_std(s_fluc)", name="s_fluc_std")
        analysis_profile.add_task("plane_avg(s_mean)", name="s_mean")
        analysis_profile.add_task("plane_avg(s_fluc + s_mean)", name="s_tot")
        analysis_profile.add_task("plane_avg(dz(s_fluc))", name="grad_s_fluc")        
        analysis_profile.add_task("plane_avg(dz(s_mean))", name="grad_s_mean")        
        analysis_profile.add_task("plane_avg(dz(s_fluc + s_mean))", name="grad_s_tot")
        analysis_profile.add_task("plane_avg(g*dz(s_fluc)*Cp_inv)", name="brunt_squared_fluc")        
        analysis_profile.add_task("plane_avg(g*dz(s_mean)*Cp_inv)", name="brunt_squared_mean")        
        analysis_profile.add_task("plane_avg(g*dz(s_fluc + s_mean)*Cp_inv)", name="brunt_squared_tot")
        
        analysis_tasks['profile'] = analysis_profile

        analysis_scalar = solver.evaluator.add_file_handler(data_dir+"scalar", max_writes=20, parallel=False,
                                                            write_num=scalar[0], set_num=scalar[1], **kwargs)
        analysis_scalar.add_task("vol_avg(KE)", name="KE")
        analysis_scalar.add_task("vol_avg(PE)", name="PE")
        analysis_scalar.add_task("vol_avg(IE)", name="IE")
        analysis_scalar.add_task("vol_avg(PE_fluc)", name="PE_fluc")
        analysis_scalar.add_task("vol_avg(IE_fluc)", name="IE_fluc")
        analysis_scalar.add_task("vol_avg(KE + PE + IE)", name="TE")
        analysis_scalar.add_task("vol_avg(KE + PE_fluc + IE_fluc)", name="TE_fluc")
        analysis_scalar.add_task("vol_avg(u_rms)", name="u_rms")
        analysis_scalar.add_task("vol_avg(w_rms)", name="w_rms")
        analysis_scalar.add_task("vol_avg(Re_rms)", name="Re_rms")
        analysis_scalar.add_task("vol_avg(Pe_rms)", name="Pe_rms")
        analysis_scalar.add_task("vol_avg(Ma_iso_rms)", name="Ma_iso")
        analysis_scalar.add_task("vol_avg(Ma_ad_rms)", name="Ma_ad")
        analysis_scalar.add_task("vol_avg(enstrophy)", name="enstrophy")
        analysis_scalar.add_task("vol_avg(Nusselt)", name="Nusselt")

        analysis_tasks['scalar'] = analysis_scalar

        return self.analysis_tasks
    
class FC_equations_2d(FC_equations):
    def __init__(self, **kwargs):
        super(FC_equations_2d, self).__init__(**kwargs)
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes'
        self.variables = ['u','u_z','w','w_z','T1', 'T1_z', 'ln_rho1']

    def _set_subs(self):
        # 2-D specific subs
        self.problem.substitutions['ω_y'] = '( u_z  - dx(w))'        
        self.problem.substitutions['enstrophy']   = '(ω_y**2)'
        self.problem.substitutions['v']           = '(0)'
        self.problem.substitutions['v_z']           = '(0)'

        # differential operators
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dz(f_z))"
        self.problem.substitutions['Div(f, f_z)'] = "(dx(f) + f_z)"
        self.problem.substitutions['Div_u'] = "Div(u, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + w*(f_z))"
        # analysis operators
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x")/Lx'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Lz'

        self.problem.substitutions["σxx"] = "(2*dx(u) - 2/3*Div_u)"
        self.problem.substitutions["σzz"] = "(2*w_z   - 2/3*Div_u)"
        self.problem.substitutions["σxz"] = "(dx(w) +  u_z )"

        super(FC_equations_2d, self)._set_subs()
        
    def set_equations(self, Rayleigh, Prandtl, kx = 0, EVP_2 = False, 
                      easy_rho_momentum=False, easy_rho_energy=False):

        if self.dimensions == 1:
            self.problem.parameters['j'] = 1j
            self.problem.substitutions['dx(f)'] = "j*kx*(f)"
            self.problem.parameters['kx'] = kx
        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
        self._set_parameters()
        if EVP_2:
            self.problem.substitutions['chi'] = "(Prandtl*nu)"
            self.problem.parameters['Prandtl'] = Prandtl
            self.problem.parameters.pop('nu')
            self.problem.parameters.pop('chi')

        self._set_subs()
        
        self.viscous_term_u = " nu*(Lap(u, u_z) + 1/3*Div(dx(u), dx(w_z)))"
        self.viscous_term_w = " nu*(Lap(w, w_z) + 1/3*Div(  u_z, dz(w_z)))"
        
        if not easy_rho_momentum:
            self.viscous_term_u += " + (nu*del_ln_rho0 + del_nu) * σxz"
            self.viscous_term_w += " + (nu*del_ln_rho0 + del_nu) * σzz"

        self.problem.substitutions['L_visc_w'] = self.viscous_term_w
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u

        self.nonlinear_viscous_u = " nu*(dx(ln_rho1)*σxx + dz(ln_rho1)*σxz)"
        self.nonlinear_viscous_w = " nu*(dx(ln_rho1)*σxz + dz(ln_rho1)*σzz)"
        
        self.problem.substitutions['NL_visc_w'] = self.nonlinear_viscous_w
        self.problem.substitutions['NL_visc_u'] = self.nonlinear_viscous_u

        # double check implementation of variabile chi and background coupling term.
        self.problem.substitutions['Q_z'] = "(-T1_z)"
        self.linear_thermal_diff    = " Cv_inv*(chi*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.nonlinear_thermal_diff = " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source =                 " Cv_inv*(chi*(T0_zz) - Qcool_z/rho_full)"
        if not easy_rho_energy:
            self.linear_thermal_diff += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T1_z'
            self.source              += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T0_z'
                
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff 
        self.problem.substitutions['NL_thermal']   = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source

        # double check this      
        self.viscous_heating = " Cv_inv*nu*(dx(u)*σxx + w_z*σzz + σxz**2)"

        self.problem.substitutions['NL_visc_heat'] = self.viscous_heating
       
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale_momentum)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + NL_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + NL_visc_u)"))


        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)")) 
                

    def initialize_output(self, solver, data_dir, full_output=False, coeffs_output=True,
                          slices=[1,1], profiles=[1,1], scalar=[1,1], coeffs=[1,1], **kwargs):
        #  slices, profiles, and scalar are all [write_num, set_num]

        analysis_tasks = super(FC_equations_2d, self).initialize_output(solver, data_dir, full_output=full_output,
                          slices=slices, profiles=profiles, scalar=scalar, coeffs=coeffs, **kwargs)
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False,
                                                           write_num=slices[0], set_num=slices[1], **kwargs)
        analysis_slice.add_task("s_fluc", name="s")
        analysis_slice.add_task("T1", name="T")
        analysis_slice.add_task("ln_rho1", name="ln_rho")
        analysis_slice.add_task("u", name="u")
        analysis_slice.add_task("w", name="w")
        analysis_slice.add_task("ω_y", name="vorticity")
        analysis_tasks['slice'] = analysis_slice

        if coeffs_output:
            analysis_coeff = solver.evaluator.add_file_handler(data_dir+"coeffs", max_writes=20, parallel=False,
                                                               write_num=coeffs[0], set_num=coeffs[1], **kwargs)
            analysis_coeff.add_task("s_fluc", name="s", layout='c')
            analysis_coeff.add_task("T1", name="T", layout='c')
            analysis_coeff.add_task("ln_rho1", name="ln_rho", layout='c')
            analysis_coeff.add_task("u", name="u", layout='c')
            analysis_coeff.add_task("w", name="w", layout='c')
            analysis_coeff.add_task("ω_y", name="vorticity", layout='c')
            analysis_tasks['coeff'] = analysis_coeff
        
        return self.analysis_tasks

class FC_equations_3d(FC_equations):
    def __init__(self, **kwargs):
        super(FC_equations_3d, self).__init__(**kwargs)
        self.equation_set = 'Fully Compressible (FC) Navier-Stokes in 3-D'
        self.variables = ['u','u_z','v','v_z','w','w_z','T1', 'T1_z', 'ln_rho1']
    
    def _set_subs(self):
        # 3-D specific subs
        self.problem.substitutions['ω_x'] = '(dy(w) - v_z)'        
        self.problem.substitutions['ω_y'] = '( u_z  - dx(w))'        
        self.problem.substitutions['ω_z'] = '(dx(v) - dy(u))'        
        self.problem.substitutions['enstrophy']   = '(ω_x**2 + ω_y**2 + ω_z**2)'

        # differential operators
        self.problem.substitutions['Lap(f, f_z)'] = "(dx(dx(f)) + dy(dy(f)) + dz(f_z))"
        self.problem.substitutions['Div(fx, fy, fz_z)'] = "(dx(fx) + dy(fy) + fz_z)"
        self.problem.substitutions['Div_u'] = "Div(u, v, w_z)"
        self.problem.substitutions['UdotGrad(f, f_z)'] = "(u*dx(f) + v*dy(f) + w*(f_z))"
                    
        # analysis operators
        self.problem.substitutions['plane_avg(A)'] = 'integ(A, "x", "y")/Lx/Ly'
        self.problem.substitutions['vol_avg(A)']   = 'integ(A)/Lx/Ly/Lz'
        self.problem.substitutions['plane_std(A)'] = 'sqrt(plane_avg((A - plane_avg(A))**2))'

        self.problem.substitutions["σxx"] = "(2*dx(u) - 2/3*Div_u)"
        self.problem.substitutions["σyy"] = "(2*dy(v) - 2/3*Div_u)"
        self.problem.substitutions["σzz"] = "(2*w_z   - 2/3*Div_u)"
        self.problem.substitutions["σxy"] = "(dx(v) + dy(u))"
        self.problem.substitutions["σxz"] = "(dx(w) +  u_z )"
        self.problem.substitutions["σyz"] = "(dy(w) +  v_z )"
           
        super(FC_equations_3d, self)._set_subs()
                
    def set_equations(self, Rayleigh, Prandtl, kx = 0, EVP_2 = False, 
                      easy_rho_momentum=False, easy_rho_energy=False):

        self._set_diffusivities(Rayleigh=Rayleigh, Prandtl=Prandtl)
        self._set_parameters()
        if EVP_2:
            self.problem.substitutions['chi'] = "(Prandtl*nu)"
            self.problem.parameters['Prandtl'] = Prandtl
            self.problem.parameters.pop('nu')
            self.problem.parameters.pop('chi')
 
        self._set_subs()
        
        # here, nu and chi are constants        
        self.viscous_term_u = " nu*(Lap(u, u_z) + 1/3*Div(dx(u), dx(v), dx(w_z)))"
        self.viscous_term_v = " nu*(Lap(v, v_z) + 1/3*Div(dy(u), dy(v), dy(w_z)))"
        self.viscous_term_w = " nu*(Lap(w, w_z) + 1/3*Div(  u_z,   v_z, dz(w_z)))"
        
        if not easy_rho_momentum:
            self.viscous_term_u += " + (nu*del_ln_rho0 + del_nu) * σxz"
            self.viscous_term_v += " + (nu*del_ln_rho0 + del_nu) * σyz"
            self.viscous_term_w += " + (nu*del_ln_rho0 + del_nu) * σzz"

        self.problem.substitutions['L_visc_w'] = self.viscous_term_w
        self.problem.substitutions['L_visc_u'] = self.viscous_term_u
        self.problem.substitutions['L_visc_v'] = self.viscous_term_v

        self.nonlinear_viscous_u = " nu*(dx(ln_rho1)*σxx + dy(ln_rho1)*σxy + dz(ln_rho1)*σxz)"
        self.nonlinear_viscous_v = " nu*(dx(ln_rho1)*σxy + dy(ln_rho1)*σyy + dz(ln_rho1)*σyz)"
        self.nonlinear_viscous_w = " nu*(dx(ln_rho1)*σxz + dy(ln_rho1)*σyz + dz(ln_rho1)*σzz)"

        self.problem.substitutions['NL_visc_u'] = self.nonlinear_viscous_u
        self.problem.substitutions['NL_visc_v'] = self.nonlinear_viscous_v
        self.problem.substitutions['NL_visc_w'] = self.nonlinear_viscous_w

        # double check implementation of variabile chi and background coupling term.
        self.problem.substitutions['Q_z'] = "(-T1_z)"
        self.linear_thermal_diff    = " Cv_inv*(chi*(Lap(T1, T1_z) + T0_z*dz(ln_rho1)))"
        self.nonlinear_thermal_diff = " Cv_inv*chi*(dx(T1)*dx(ln_rho1) + dy(T1)*dy(ln_rho1) + T1_z*dz(ln_rho1))"
        self.source =                 " Cv_inv*(chi*(T0_zz) - Qcool_z/rho_full)"
        if not easy_rho_energy:
            self.linear_thermal_diff += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T1_z'
            self.source              += '+ Cv_inv*(chi*del_ln_rho0 + del_chi)*T0_z'
                
        self.problem.substitutions['L_thermal']    = self.linear_thermal_diff 
        self.problem.substitutions['NL_thermal']   = self.nonlinear_thermal_diff
        self.problem.substitutions['source_terms'] = self.source

        # check if these are the same.
        #self.viscous_heating = " Cv_inv*nu*(2*(dx(u))**2 + (dx(w))**2 + u_z**2 + 2*w_z**2 + 2*u_z*dx(w) - 2/3*Div_u**2)"
        self.viscous_heating = " Cv_inv*nu*(dx(u)*σxx + dy(v)*σyy + w_z*σzz + σxy**2 + σxz**2 + σyz**2)"

        self.problem.substitutions['NL_visc_heat'] = self.viscous_heating
       
        self.problem.add_equation("dz(u) - u_z = 0")
        self.problem.add_equation("dz(v) - v_z = 0")
        self.problem.add_equation("dz(w) - w_z = 0")
        self.problem.add_equation("dz(T1) - T1_z = 0")

        logger.debug("Setting continuity equation")
        self.problem.add_equation(("(scale_continuity)*( dt(ln_rho1)   + w*del_ln_rho0 + Div_u ) = "
                                   "(scale_continuity)*(-UdotGrad(ln_rho1, dz(ln_rho1)))"))

        logger.debug("Setting z-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(w) + T1_z   + T0*dz(ln_rho1) + T1*del_ln_rho0 - L_visc_w) = "
                                   "(scale_momentum)*(-T1*dz(ln_rho1) - UdotGrad(w, w_z) + NL_visc_w)"))
        
        logger.debug("Setting x-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(u) + dx(T1) + T0*dx(ln_rho1)                  - L_visc_u) = "
                                   "(scale_momentum)*(-T1*dx(ln_rho1) - UdotGrad(u, u_z) + NL_visc_u)"))

        logger.debug("Setting y-momentum equation")
        self.problem.add_equation(("(scale_momentum)*( dt(v) + dy(T1) + T0*dy(ln_rho1)                  - L_visc_v) = "
                                   "(scale_momentum)*(-T1*dy(ln_rho1) - UdotGrad(v, v_z) + NL_visc_v)"))

        logger.debug("Setting energy equation")
        self.problem.add_equation(("(scale_energy)*( dt(T1)   + w*T0_z + (gamma-1)*T0*Div_u -  L_thermal) = "
                                   "(scale_energy)*(-UdotGrad(T1, T1_z)    - (gamma-1)*T1*Div_u + NL_thermal + NL_visc_heat + source_terms)"))
        

    def set_BC(self, **kwargs):        
        super(FC_equations_3d, self).set_BC(**kwargs)
        # stress free boundary conditions.
        self.problem.add_bc("left(v_z) = 0")
        self.problem.add_bc("right(v_z) = 0")
        self.dirichlet_set.append('v_z')
        for key in self.dirichlet_set:
            self.problem.meta[key]['z']['dirichlet'] = True

        
    def initialize_output(self, solver, data_dir, full_output=False,
                          slices=[1,1], profiles=[1,1], scalar=[1,1], coeffs=[1,1], **kwargs):
        #  slices, profiles, and scalar are all [write_num, set_num]

        analysis_tasks = super(FC_equations_3d, self).initialize_output(solver, data_dir, full_output=full_output,
                          slices=slices, profiles=profiles, scalar=scalar, coeffs=coeffs, **kwargs)
        
        analysis_slice = solver.evaluator.add_file_handler(data_dir+"slices", max_writes=20, parallel=False,
                                                           write_num=slices[0], set_num=slices[1], **kwargs)
        analysis_slice.add_task("interp(s_fluc,                     y={})".format(self.Ly/2), name="s")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), y={})".format(self.Ly/2), name="s'")
        analysis_slice.add_task("interp(enstrophy,                  y={})".format(self.Ly/2), name="enstrophy")
        analysis_slice.add_task("interp(ω_y,                        y={})".format(self.Ly/2), name="vorticity")
        analysis_slice.add_task("interp(s_fluc,                     z={})".format(0.95*self.Lz), name="s near top")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), z={})".format(0.95*self.Lz), name="s' near top")
        analysis_slice.add_task("interp(enstrophy,                  z={})".format(0.95*self.Lz), name="enstrophy near top")
        analysis_slice.add_task("interp(ω_z,                        z={})".format(0.95*self.Lz), name="vorticity_z near top")
        analysis_slice.add_task("interp(s_fluc,                     z={})".format(0.5*self.Lz),  name="s midplane")
        analysis_slice.add_task("interp(s_fluc - plane_avg(s_fluc), z={})".format(0.5*self.Lz),  name="s' midplane")
        analysis_slice.add_task("interp(enstrophy,                  z={})".format(0.5*self.Lz),  name="enstrophy midplane")
        analysis_slice.add_task("interp(ω_z,                        z={})".format(0.5*self.Lz),  name="vorticity_z midplane")
        analysis_tasks['slice'] = analysis_slice
        return self.analysis_tasks
            
class FC_polytrope_2d(FC_equations_2d, Polytrope):
    def __init__(self, dimensions=2, *args, **kwargs):
        super(FC_polytrope_2d, self).__init__(dimensions=dimensions) 
        Polytrope.__init__(self, dimensions=dimensions, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        easy_rho_momentum, easy_rho_energy = False, False
        if self.constant_mu:
            easy_rho_momentum   = True
        if self.constant_kappa:
            easy_rho_energy     = True
        #Need to implement HS equilibrium keyword and equation stuff
        super(FC_polytrope_2d, self).set_equations(*args,  easy_rho_momentum = easy_rho_momentum,
                                                        easy_rho_energy   = easy_rho_energy,
                                                        **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)
    
    def initialize_output(self, solver, data_dir, *args, **kwargs):
        super(FC_polytrope_2d, self).initialize_output(solver, data_dir, *args, **kwargs)
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
        for key in self.problem.parameters.keys():
            if 'scale' in key:
                continue
            if type(self.problem.parameters[key]) == Field:
                self.problem.parameters[key].set_scales(1, keep_data=True)
                this_chunk      = np.zeros(self.nz)
                global_chunk    = np.zeros(self.nz)
                n_per_cpu       = int(self.nz/self.domain.dist.comm_cart.size)
                this_chunk[ self.domain.dist.comm_cart.rank*(n_per_cpu):\
                            (self.domain.dist.comm_cart.rank+1)*(n_per_cpu)] = \
                                    self.problem.parameters[key]['g'][0,:]
                self.domain.dist.comm_cart.Allreduce(this_chunk, global_chunk, op=MPI.SUM)
                if self.domain.dist.comm_cart.rank == 0:
                    f[key] = global_chunk
            elif self.domain.dist.comm_cart.rank == 0:
                f[key] = self.problem.parameters[key]
        if self.domain.dist.comm_cart.rank == 0:
            f['dimensions']     = 2
            f['nx']             = self.nx
            f['nz']             = self.nz
            f['m_ad']           = self.m_ad
            f['m']              = self.m_ad - self.epsilon
            f['epsilon']        = self.epsilon
            f['n_rho_cz']       = self.n_rho_cz
            f['rayleigh']       = self.Rayleigh
            f['prandtl']        = self.Prandtl
            f['aspect_ratio']   = self.aspect_ratio
            f['atmosphere_name']= self.atmosphere_name
            f.close()

class FC_polytrope_3d(FC_equations_3d, Polytrope):
    def __init__(self, dimensions=3, *args, **kwargs):
        super(FC_polytrope_3d, self).__init__(dimensions=dimensions) 
        Polytrope.__init__(self, dimensions=dimensions, *args, **kwargs)
        logger.info("solving {} in a {} atmosphere".format(self.equation_set, self.atmosphere_name))

    def set_equations(self, *args, **kwargs):
        easy_rho_momentum, easy_rho_energy = False, False
        if self.constant_mu:
            easy_rho_momentum   = True
        if self.constant_kappa:
            easy_rho_energy     = True
        super(FC_polytrope_3d, self).set_equations(*args,  easy_rho_momentum = easy_rho_momentum,
                                                        easy_rho_energy   = easy_rho_energy,
                                                        **kwargs)
        self.test_hydrostatic_balance(T=self.T0, rho=self.rho0)

