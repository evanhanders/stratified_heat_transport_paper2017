#/usr/bin/env python
import os
import sys
import time
import warnings
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import dedalus.public as de
import h5py
from mpi4py import MPI
from scipy import optimize as opt
from scipy import interpolate


from tools.eigentools.eigentools import Eigenproblem, CriticalFinder
from stratified_dynamics import polytropes, multitropes


CW = MPI.COMM_WORLD
warnings.filterwarnings("ignore")

class OnsetSolver:
    """
    This class finds the onset of convection in a specified atmosphere
    (currently multitropes and polytropes) using a specified equation
    set (currently FC Navier Stokes is implemented).

    NOTE: This class currently depends on Evan Anders' branch of eigentools,
    found at https://bitbucket.org/evanhanders/eigentools.  Enter the local
    tools/ directory of your polytrope repo and clone eigentools there.
    """

    def __init__(self, eqn_set=0, atmosphere=0, ra_steps=(1, 1e3, 40, True),
                 kx_steps=(0.01, 1, 40, True), ky_steps=None,
                 atmo_kwargs={}, eqn_args=[], eqn_kwargs={}, bc_kwargs={}):
        """
        Initializes the onset solver by specifying the equation set to be used
        and the type of atmosphere that will be solved on.  Also specifies
        The range of Rayleigh numbers and horizontal wavenumbers to examine.

        Keyword Arguments:
            eqn_set     - An integer, specifying the equation set to solve.
                            Options:
                             (0) FC Hydro
            atmosphere  - An integer, specifying the type of atmosphere.
                            Options:
                             (0) Polytrope
                             (1) Multitrope
            ra_steps    - A tuple containing four elements:
                            1. Min Ra to solve eigenproblem at
                            2. Max Ra to solve eigenproblem at
                            3. Num steps in Ra space between min/max
                            4. A bool.  If True, step through Ra in log space,
                               if False, step through in linear space.
            kx_steps    - A tuple containing four elements.  All elements are
                          the same as in ra_steps, just for horizontal
                          wavenumber.  Note here that kx=1 means that the
                          horizontal wavemode being examined is the wavemode
                          corresponding to a wavelength of the domain depth.
            ky_steps    - Same as kx_steps for the y direction.  If None, 2D
                          eqns will be used.
            atmo_kwargs - A Python dictionary, containing all default
                          information for the atmosphere, including 
                          number of z points, superadiabaticity, etc.
            eqn_args     - A list of arguments to be passed to set_equations
            eqn_kwargs   - A dictionary of keyword arguments to be passed to 
                           set_equations
            bc_kwargs    - A list of keyword arguments to be passed to 
                           set_BC
        """
        self._eqn_set    = eqn_set
        self._atmosphere = atmosphere
        self._ra_steps   = ra_steps
        self._kx_steps   = kx_steps
        self._ky_steps   = ky_steps

        self._atmo_kwargs = atmo_kwargs
        self._eqn_args    = eqn_args
        self._eqn_kwargs  = eqn_kwargs
        self._bc_kwargs   = bc_kwargs
        self.cf = CriticalFinder(self.solve_problem, CW)

    def _find_3D_mins(kx_curves, ky_curves, ra_curves, n_pts=pts_per_curve):
        f_interp = interpolate.interp2d(kx_curves.flatten(), ky_curves.flatten(), ra_curves.flatten())

        kxs = np.logspace(np.log10(np.min(kx_curves)), np.log10(np.max(kx_curves)), n_pts)
        kys = np.logspace(np.log10(np.min(ky_curves)), np.log10(np.max(ky_curves)), n_pts)

        xxs, yys = np.meshgrid(kxs, kys)
        ras = f_interp(kxs, kys)
        where = np.argmin(ras)

        return xxs, yys, ras, ras[where], xxs[where], yys[where]

    def _grid_to_onset_curve(self, n_pts=1000):
        """
        Look in the CriticalFinder instance inside of this instance, pull out
        zero-values of growth as a function of kx, ra.  Interpolate that curve
        of kx, ra to find an onset curve & return that curve.

        Keyword Arguments:
            n_pts   - The number of data points to put in the interpolated
                      onset curve.  More = smoother, to a point.
        """
        self.cf.root_finder()
        mask = np.isfinite(self.cf.roots)
        kxs_roots = self.cf.yy[mask, 0]
        ras_roots = self.cf.roots[mask]
        kxs = np.logspace(np.log10(np.ma.min(kxs_roots)), 
                          np.log10(np.ma.max(kxs_roots)), 
                          n_pts)
        ras = np.interp(kxs, kxs_roots, ras_roots)

        return ras, kxs

    def _initialize_output_2D(self, out_dir, out_file_name, pts_per_curve=1000):
        """
        creates a .h5 file in which all output can be properly stored.

        Arguments:
            out_dir         - The directory for outputting the file.
            out_file_name   - The name of the file to save info into.  If
                              None, make a filename based on the date.
        """
        if self.cf.comm.rank != 0:
            return
        if out_file_name == None:
            import time
            now = time.strftime('%Y_%m_%d_%H-%M-%S')
            out_file_name = 'onset_curves_{:s}.h5'.format(now)
        
        self.save_file = '{:s}/{:s}'.format(out_dir, out_file_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        f = h5py.File(self.save_file, 'w')
        

        if not hasattr(self, '_tasks'):
            f.create_dataset('ra_curve', (pts_per_curve,), dtype=np.float64)
            f.create_dataset('kx_curve', (pts_per_curve,), dtype=np.float64)
            f.create_dataset('ra_crit', (1,), dtype=np.float64)
            f.create_dataset('kx_crit', (1,), dtype=np.float64)
            f.create_dataset('kx_conv', (1,), dtype=np.float64)
        else:
            for key in self._tasks.keys():
                task = self._tasks[key]

                f.create_dataset('{:s}_vals'.format(key), 
                                 (task[2],), dtype=np.float64)
                f.create_dataset('{:s}_ra_curves'.format(key),
                                 (task[2],pts_per_curve), dtype=np.float64)
                f.create_dataset('{:s}_kx_curves'.format(key),
                                 (task[2],pts_per_curve), dtype=np.float64)
                f.create_dataset('{:s}_ra_crits'.format(key),
                                 (task[2],), dtype=np.float64)
                f.create_dataset('{:s}_kx_crits'.format(key),
                                 (task[2],), dtype=np.float64)
                f.create_dataset('{:s}_kx_convs'.format(key),
                                 (task[2],), dtype=np.float64)

        f.close()

    def _initialize_output_3D(self, out_dir, out_file_name, pts_per_curve=100):
        """
        creates a .h5 file in which all output can be properly stored.

        Arguments:
            out_dir         - The directory for outputting the file.
            out_file_name   - The name of the file to save info into.  If
                              None, make a filename based on the date.
        """
        if self.cf.comm.rank != 0:
            return
        if out_file_name == None:
            import time
            now = time.strftime('%Y_%m_%d_%H-%M-%S')
            out_file_name = 'onset_curves_{:s}.h5'.format(now)
        
        self.save_file = '{:s}/{:s}'.format(out_dir, out_file_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        f = h5py.File(self.save_file, 'w')
        

        if not hasattr(self, '_tasks'):
            f.create_dataset('ra_curve', (pts_per_curve,pts_per_curve), 
                             dtype=np.float64)
            f.create_dataset('kx_curve', (pts_per_curve,pts_per_curve), 
                             dtype=np.float64)
            f.create_dataset('ky_curve', (pts_per_curve,pts_per_curve), 
                             dtype=np.float64)
            f.create_dataset('ra_crit',  (1,), dtype=np.float64)
            f.create_dataset('kx_crit',  (1,), dtype=np.float64)
            f.create_dataset('ky_crit',  (1,), dtype=np.float64)
            f.create_dataset('kx_conv',  (1,), dtype=np.float64)
        else:
            for key in self._tasks.keys():
                task = self._tasks[key]

                f.create_dataset('{:s}_vals'.format(key), 
                                 (task[2],), dtype=np.float64)
                f.create_dataset('{:s}_ra_curves'.format(key),
                                 (task[2], pts_per_curve, pts_per_curve), 
                                 dtype=np.float64)
                f.create_dataset('{:s}_kx_curves'.format(key),
                                 (task[2], pts_per_curve, pts_per_curve), 
                                 dtype=np.float64)
                f.create_dataset('{:s}_ky_curves'.format(key),
                                 (task[2], pts_per_curve, pts_per_curve), 
                                 dtype=np.float64)
                f.create_dataset('{:s}_ra_crits'.format(key),
                                 (task[2],), dtype=np.float64)
                f.create_dataset('{:s}_kx_crits'.format(key),
                                 (task[2],), dtype=np.float64)
                f.create_dataset('{:s}_ky_crits'.format(key),
                                 (task[2],), dtype=np.float64)
                f.create_dataset('{:s}_kx_convs'.format(key),
                                 (task[2],), dtype=np.float64)

        f.close()



    def _save_set_2D(self, ra_curve, kx_curve, ra_crit, kx_crit, kx_conv,\
                    key=None, val=None, num_write=0):
        """
        Saves an onset curve.  Called automatically by the find_crits() function
        after each critical value is found.  This ensures that, even if not ALL
        cases finish, not all information is lost.
        """
        if self.cf.comm.rank != 0:
            return

        f = h5py.File(self.save_file, 'r+')
       
        #If there aren't any tasks, just save this curve
        if key == None:
            data_keys = ('ra_curve', 'kx_curve', 'ra_crit', 'kx_crit', 'kx_conv')
            data = (ra_curve, kx_curve, ra_crit, kx_crit, kx_conv)
            for i,d_key in enumerate(data_keys):
                dataset = f[d_key]
                dataset[:] = data[i]
        else:
            #Save this according to its task
            scalar_keys = ('{:s}_vals'.format(key), '{:s}_ra_crits'.format(key),\
                           '{:s}_kx_crits'.format(key), '{:s}_kx_convs'.format(key))
            scalar_data = (val, ra_crit, kx_crit, kx_conv)
            for i, d_key in enumerate(scalar_keys):
                dataset = f[d_key]
                dataset[num_write] = scalar_data[i]

            array_keys = ('{:s}_ra_curves'.format(key), '{:s}_kx_curves'.format(key))
            array_data = (ra_curve, kx_curve)
            for i, d_key in enumerate(array_keys):
                dataset = f[d_key]
                dataset[num_write, :] = array_data[i]
        f.close()

    def _save_set_3D(self, ra_curve, kx_curve, ky_curve, ra_crit, kx_crit, ky_crit, kx_conv,\
                    key=None, val=None, num_write=0):
        """
        Saves an onset curve.  Called automatically by the find_crits() function
        after each critical value is found.  This ensures that, even if not ALL
        cases finish, not all information is lost.
        """
        if self.cf.comm.rank != 0:
            return

        f = h5py.File(self.save_file, 'r+')
       
        #If there aren't any tasks, just save this curve
        if key == None:
            data_keys = ('ra_curve', 'kx_curve', 'ky_curve', 'ra_crit', 'kx_crit', 'ky_crit', 'kx_conv')
            data = (ra_curve, kx_curve, ky_curve, ra_crit, kx_crit, ky_crit, kx_conv)
            for i,d_key in enumerate(data_keys):
                dataset = f[d_key]
                dataset[:] = data[i]
        else:
            #Save this according to its task
            scalar_keys = ('{:s}_vals'.format(key), '{:s}_ra_crits'.format(key),\
                           '{:s}_kx_crits'.format(key), '{:s}_ky_crits'.format(key),\
                           '{:s}_kx_convs'.format(key))
            scalar_data = (val, ra_crit, kx_crit, ky_crit, kx_conv)
            for i, d_key in enumerate(scalar_keys):
                dataset = f[d_key]
                dataset[num_write] = scalar_data[i]

            array_keys = ('{:s}_ra_curves'.format(key), '{:s}_kx_curves'.format(key),
                          '{:s}_ky_curves'.format(key) )
            array_data = (ra_curve, kx_curve)
            for i, d_key in enumerate(array_keys):
                dataset = f[d_key]
                dataset[num_write, :,:] = array_data[i]
        f.close()
        

    def add_task(self, name, min_val, max_val, n_steps=10, log=False):
        """
        Adds a dimension to the eigenvalue solve.  Allows examination of
        onset for different values of atmospheric parameters than those
        specified in set_defaults.

        Arguments:
            name        - A string containing the name of the atmospheric
                          parameter that is being varied.  For example,
                          'n_rho_cz' or 'epsilon'
            min_val     - The minimum value of this parameter to examine
            max_val     - The maximum value of this parameter to examine
        
        Keyword Arguments:
            n_steps     - The number of steps in parameter space to take
                          between min_val and max_val
            log         - If True, step through log_10(parameter) space,
                          otherwise step through parameter space linearly.
        """
        if not hasattr(self, '_tasks'):
            self._tasks = dict()

        self._tasks[name] = (min_val, max_val, n_steps, log)
 
    def find_crits(self, tol=1e-3, pts_per_curve=1000, 
                   out_dir='./', out_file=None):
        """
        Steps through all tasks and solves eigenvalue problems for
        the specified parameters.  If no tasks are specified, only
        the default parameters are solved for over the given kx/ra
        range.

        Keyword Arguments:
            tol             - The convergence tolerance for the iterative crit
                              finder (e.g., how little can the answer change
                              between steps before we're happy)
            pts_per_curve   - # of points to use on interpolated critical curve
            out_dir         - Output directory of information files
            out_file        - Name of information file.  If None, auto generate.
        """

        if type(self._ky_steps) == type(None):
            self._initialize_output_2D(out_dir, out_file, pts_per_curve=pts_per_curve)
        else:
            self._initialize_output_3D(out_dir, out_file, pts_per_curve=pts_per_curve)
        self._ky_adjust = False

        self._data = dict()
        if not hasattr(self, '_tasks'):
            if type(self._ky_steps) == type(None):
                self.cf = CriticalFinder(self.solve_problem, CW)
                # If no tasks specified, set the atmospheric defaults,
                # find the crits, and store the curves
                self.atmo_kwargs = self._atmo_kwargs
                ra_crit, kx_crit = self.cf.iterative_crit_finder(
                                self._ra_steps[0], self._ra_steps[1],
                                self._kx_steps[0], self._kx_steps[1],
                                self._ra_steps[2], self._kx_steps[2],
                                log_x = self._ra_steps[3], 
                                log_y = self._kx_steps[3], tol=tol)
                ra_curve, kx_curve = self._grid_to_onset_curve(n_pts=pts_per_curve)
                self._save_set_2D(ra_curve, kx_curve, 
                               ra_crit, kx_crit, 2*np.pi/self.atmosphere.Lz)
                self._data['ra_curve'] = ra_curve
                self._data['kx_curve'] = kx_curve
                self._data['ra_crit']  = ra_crit
                self._data['kx_crit']  = kx_crit
                self._data['kx_conv']  = 2*np.pi/self.atmosphere.Lz
            else:
                if self._ky_steps[-1]:
                    ky_steps = np.logspace(np.log10(self._ky_steps[0]), 
                                           np.log10(self._ky_steps[1]), 
                                           self._ky_steps[2])
                else:
                    ky_steps = np.linspace(self._ky_steps[0],
                                           self._ky_steps[1],
                                           self._ky_steps[2])
                kx_curves = np.zeros((self._ky_steps[2], pts_per_curve))
                ky_curves = np.zeros((self._ky_steps[2], pts_per_curve))
                ra_curves = np.zeros_like(kx_curves)
                kx_crits = np.zeros(self._ky_steps[2])
                ra_crits = np.zeros_like(kx_crits)
                for i,ky in enumerate(ky_steps):
                    ky_curves[i,:] = ky
                    self._eqn_kwargs['ky'] = ky
                    self._ky_adjust = True
                    self.cf = CriticalFinder(self.solve_problem, CW)
                    # If no tasks specified, set the atmospheric defaults,
                    # find the crits, and store the curves
                    self.atmo_kwargs = self._atmo_kwargs
                    ra_crits[i], kx_crits[i] = self.cf.iterative_crit_finder(
                                    self._ra_steps[0], self._ra_steps[1],
                                    self._kx_steps[0], self._kx_steps[1],
                                    self._ra_steps[2], self._kx_steps[2],
                                    log_x = self._ra_steps[3], 
                                    log_y = self._kx_steps[3], tol=tol)
                    ra_curves[i,:], kx_curves[i,:] = self._grid_to_onset_curve(n_pts=pts_per_curve)

                kx_curves, ky_curves, ra_curves, min_kx, min_ky, min_ra = 
                        self._find_3D_mins(kx_curves, ky_curves, ra_curves, n_pts=pts_per_curve)
                
                self._save_set_3D(ra_curves, kx_curves, ky_curves, 
                               min_ra, min_kx, min_ky, 2*np.pi/self.atmosphere.Lz)
                self._data['ra_curves'] = ra_curves
                self._data['kx_curves'] = kx_curves
                self._data['ky_curves'] = ky_curves
                self._data['ra_crit']  = min_ra
                self._data['kx_crit']  = min_kx
                self._data['ky_crit']  = min_ky
                self._data['kx_conv']  = 2*np.pi/self.atmosphere.Lz
        else:
            #Loop through each task
            for key in self._tasks.keys():
                task = self._tasks[key]
                #Start with the defaults
                self.atmo_kwargs = self._atmo_kwargs
                onset_kx_curves = []
                onset_ra_curves = []
                onset_kx_vals   = []
                onset_ra_vals   = []
                kx_conversion   = []
                if task[-1]:
                    values = np.logspace(
                                np.log10(task[0]), np.log10(task[1]), task[2])
                else:
                    values = np.linspace(task[0], task[1], task[2])
                for i,value in enumerate(values):
                    self.cf = CriticalFinder(self.solve_problem, CW)
                    # For each value in parameter space, overwrite that info
                    # compared to the defaults. Solve.  Store onset curve.
                    self.atmo_kwargs[key] = value
                    ra_crit, kx_crit = self.cf.iterative_crit_finder(
                                    self._ra_steps[0], self._ra_steps[1],
                                    self._kx_steps[0], self._kx_steps[1],
                                    self._ra_steps[2], self._kx_steps[2],
                                    log_x = self._ra_steps[3], 
                                    log_y = self._kx_steps[3], tol=tol)
                    ra_curve, kx_curve = self._grid_to_onset_curve(
                                                    n_pts=pts_per_curve)
                    self._save_set(ra_curve, kx_curve, 
                                   ra_crit, kx_crit, 2*np.pi/self.atmosphere.Lz,
                                   key=key, val=value, num_write=i)
                    onset_ra_curves.append(ra_curve)
                    onset_kx_curves.append(kx_curve)
                    onset_ra_vals.append(ra_crit)
                    onset_kx_vals.append(kx_crit)
                    kx_conversion.append(2*np.pi/self.atmosphere.Lz)
                self._data['{:s}_vals'.format(key)] = values
                self._data['{:s}_ra_curves'.format(key)] = \
                                                np.array(onset_ra_curves)
                self._data['{:s}_kx_curves'.format(key)] = \
                                                np.array(onset_kx_curves)
                self._data['{:s}_ra_crits'.format(key)]  = \
                                                np.array(onset_ra_vals)
                self._data['{:s}_kx_crits'.format(key)]  = \
                                                np.array(onset_kx_vals)
                self._data['{:s}_kx_conv'.format(key)]   = \
                                                np.array(kx_conversion)

    def plot_onset_curves(self, out_dir='./', fig_name=None, dpi=400):
        """
        Plots all Ra vs. kx curves for all defined tasks.
        Also plots Ra vs. task space for each task.
        """
        if self.cf.comm.rank != 0:
            return
        if fig_name == None:
            import time
            now = time.strftime('%Y_%m_%d_%H-%M-%S')
            fig_name = 'plotted_onsets_{:s}.png'.format(now)
        
        fig_path = '{:s}/{:s}'.format(out_dir, fig_name)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)


        if not hasattr(self, '_tasks'):
            fig = plt.figure(figsize=(8,5))
            ax  = fig.add_subplot(1,1,1)
            if type(self._ky_steps) == type(None):
                bx  = ax.twiny()
                ax.plot(self._data['kx_curve']*self._data['kx_conv'], 
                        self._data['ra_curve'], 
                        label='min: {:4f}'.format(self._data['ra_crit']))
                bx.plot(self._data['kx_curve'], self._data['ra_curve'])
                ax.set_xlim(np.min(self._data['kx_curve']*self._data['kx_conv']), 
                            np.max(self._data['kx_curve']*self._data['kx_conv']))
                bx.set_xlim(np.min(self._data['kx_curve']), 
                            np.max(self._data['kx_curve']))
                ax.set_xlabel(r'$\mathrm{wavenumber}$ ($k_x$)')
                bx.set_xlabel(r'1/ $\mathrm{Aspect Ratio}$')

                for axis in [ax, bx]:
                    axis.set_ylabel(r'$\mathrm{Ra}_{\mathrm{crit}}$')
                    axis.set_xscale('log')
                    axis.set_yscale('log')
                ax.legend()
            else:
                img = ax.pcolormesh(self._data['kx_curves'], self._data['ky_curves'], 
                                self._data['ra_curves'], 
                                norm=matplotlib.colors.LogNorm(vmin=np.min(self._data['ra_curves']),
                                                               vmax=np.max(self._data['ra_curves'])),
                                cmap='Greens_r')
                ax.set_xlabel(r'$\mathrm{kx}\cdot L_z/2\pi$')
                ax.set_ylabel(r'$\mathrm{ky}\cdot L_z/2\pi$')
                ax.set_yscale('log')
                ax.set_xscale('log')
                cax, kw = matplotlib.colorbar.make_axes(ax, fraction=0.15, anchor=(0,0), location='top')
                cbar = matplotlib.colorbar.colorbar_factory(cax, img, **kw)
                trans = ax.get_xaxis_transform()
                cax.annotate('Min: {:3.4g}'.format(
                                   np.min(self._data['ra_crits']\
                                   [np.where(self._data['ra_crits'] > 0)])), 
                             (0.1, 0.01), xycoords=trans)
        else:
            n_tasks = len(self._tasks.keys())
            fig = plt.figure(figsize=(8, 5*n_tasks))
            for i, key in enumerate(self._tasks.keys()):
                task = self._tasks[key]
                ax = fig.add_subplot(n_tasks, 2, 2*i+1)
                bx = fig.add_subplot(n_tasks, 2, 2*(i+1))
                n_pts = self._data['{:s}_vals'.format(key)].shape[0]
                vals = self._data['{:s}_vals'.format(key)]
                kxs = self._data['{:s}_kx_curves'.format(key)]
                ras = self._data['{:s}_ra_curves'.format(key)]
                kx_crits = self._data['{:s}_kx_crits'.format(key)]
                ra_crits = self._data['{:s}_ra_crits'.format(key)]
                kx_convs = self._data['{:s}_kx_conv'.format(key)]

                for j in range(n_pts):
                    ax.plot(kxs[j,:], ras[j,:],
                            label='val {:1.2g} min: {:1.2g}'.format(vals[j], ra_crits[j]))
                ax.set_xlim(np.min(kxs), 
                            np.max(kxs))

                ax.set_xlabel(r'1/ $\mathrm{Aspect Ratio}$')
                ax.set_ylabel(r'$\mathrm{Ra}_{\mathrm{crit}}$')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.legend()
                 
                bx.plot(vals, ra_crits)
                bx.plot(vals, ra_crits, 'o')
                bx.set_xlabel('{:s}'.format(key))
                bx.set_ylabel(r'$\mathrm{Ra}_{\mathrm{crit}}$')
                bx.set_yscale('log')
                if task[-1]:
                    bx.set_xscale('log')

        print('saving: {:s}'.format(fig_path))
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()
       
    def solve_problem(self, kx, ra):
        """
        Given a horizontal wavenumber and Rayleigh number, create the specified
        atmosphere, solve an eigenvalue problem, and return information about 
        the growth rate of the solution.

        Arguments:
            kx  - The horizontal wavenumber, in units of 2*pi/Lz, where Lz is the
                  depth of the atmosphere.
            ra  - The Rayleigh number to be used in solving the atmosphere.
        """

        #Initialize atmosphere
        if self._eqn_set == 0:
            if type(self._ky_steps) == type(None):
                if self._atmosphere == 0:
                    self.atmosphere = polytropes.FC_polytrope_2d(
                                       dimensions=1, comm=MPI.COMM_SELF, 
                                       grid_dtype=np.complex128, **self.atmo_kwargs)
                elif self._atmosphere == 1:
                    self.atmosphere = multitropes.FC_multitrope(
                                       dimensions=1, comm=MPI.COMM_SELF, 
                                       grid_dtype=np.complex128, **self.atmo_kwargs)
            else:
                if self._atmosphere == 0:
                    self.atmosphere = polytropes.FC_polytrope_3d(
                                       dimensions=1, comm=MPI.COMM_SELF, 
                                       grid_dtype=np.complex128, **self.atmo_kwargs)
                elif self._atmosphere == 1:
                    self.atmosphere = multitropes.FC_multitrope_3d(
                                       dimensions=1, comm=MPI.COMM_SELF, 
                                       grid_dtype=np.complex128, **self.atmo_kwargs)
                
        k = kx*2*np.pi/self.atmosphere.Lz
        if 'ky' in self._eqn_kwargs:
            if self.cf.rank == 0:
                print('Solving for ky = {}'.format(self._eqn_kwargs['ky']))
            if self._ky_adjust:
                self._eqn_kwargs['ky'] *= 2*np.pi/self.atmosphere.Lz
                self._ky_adjust = False

        #Set the eigenvalue problem using the atmosphere
        self.atmosphere.set_eigenvalue_problem(ra, 
                *self._eqn_args, kx=k, **self._eqn_kwargs, tol=1e-6)
        self.atmosphere.set_BC(**self._bc_kwargs)
        problem = self.atmosphere.get_problem()

        #Solve using eigentools Eigenproblem
        self.eigprob = Eigenproblem(problem)
        max_val = self.eigprob.growth_rate({}, tol=1e-6)

        if max_val[-1] != None:
            return max_val
        else:
            return [np.nan, ]

if __name__ == '__main__':
    help(OnsetSolver)
