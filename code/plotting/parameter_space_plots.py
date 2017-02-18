"""
Script for plotting a parameter space study of Nu v Ra.

Usage:
    parameter_space_plots.py --calculate
    parameter_space_plots.py

Options:
    --calculate     If flagged, touch dedalus output files and do time averages.  If not, use post-processed data from before
"""


import matplotlib   
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 11})
import matplotlib.pyplot as plt
from base.plot_buddy import ProfileBuddy, COLORS
import os
import numpy as np
import glob
from docopt import docopt

import dedalus.public as de


COLORS=['orange', 'green', 'indigo', 'black']
MARKERS=['s', 'd', '*', 'o']

golden_ratio=1.618
FIGWIDTH=3+7.0/16
FIGHEIGHT=(FIGWIDTH/golden_ratio)*2.25 #2 plots, with padding for x-label.

#MARKERS=['s', 'o', '*']

from mpi4py import MPI

comm = MPI.COMM_WORLD

root_dir='/nobackup/eanders/sp2017/fc_poly_hydro/'
mach_dir='/nobackup/eanders/sp2017/mach_eps/'

start_file=50
n_files=1000

#change to 5
param_keys = ['Ra', 'eps', 'nrhocz', 'Pr', 'dims']
keys = ['Nusselt', 'Nusselt_5', 'Nusselt_6', 'norm_5_kappa_flux_z',  'Re_rms', 'Ma_ad', 'Ma_iso', 'ln_rho1', 'T1', 'vel_rms',\
    'rho_full', 'T_full', 'kappa_flux_z', 'enthalpy_flux_z', 'PE_flux_z', 'KE_flux_z', 'viscous_flux_z', 'kappa_flux_fluc_z']
out_dir_name = '/parameter_profiles/'
out_file_name= 'parameter'

def get_info(start_dir, keys, out_dir_name, out_file_name):
    #Get all of the directories and get info on them.
    dirs = []

    for i, dir in enumerate(glob.glob('{:s}/*/'.format(start_dir))):
        tag = dir.split(start_dir)[-1]
        info = dict()
        for stem in tag.split('_'):
            if 'eps' in stem:
                info['eps'] = float(stem.split('eps')[-1])
            if 'Ra' in stem:
                info['Ra']  = float(stem.split('Ra')[-1])
            if 'nrhocz' in stem:
                info['nrhocz'] = float(stem.split('nrhocz')[-1])
            if 'Pr' in stem:
                info['Pr']  = float(stem.split('Pr')[-1])
            if '2D' in stem:
                info['dimensions'] = 2
            elif '3D' in stem:
                info['dimensions'] = 3
        try:
            dirs.append([dir, info['eps'], info['Ra'], info['Pr'], info['nrhocz'], info['dimensions'], dict()])
        except:
            print('not a data dir')


    args = docopt(__doc__)
    ########### Analyze raw data ########################
    if args['--calculate']:
        for i, dir in enumerate(dirs):
            if np.mod(i, comm.size) != comm.rank:
                continue
            try:
    #        if True:
                comm_new = comm.Create(comm.Get_group().Incl(np.ones(1)*comm.rank))
                
                plotter = ProfileBuddy(dir[0], max_files=n_files, start_file=start_file, \
                                        write_cadence=1e10, file_dirs=['scalar', 'profiles'], comm=comm_new,
                                        outdir=out_dir_name)
                for j, key in enumerate(keys):
                    plotter.add_subplot(key, 0, j)
                plotter.analyze_subplots()
                plotter.communicate_profiles()
                plotter.save_profiles(filename=out_file_name)
                plotter.make_plots(figsize=(3*len(keys), 8))
            except:
                print('AN ERROR HAS OCCURED IN {:s}'.format(dir[0]))
    return dirs

dirs_root = get_info(root_dir, keys, out_dir_name, out_file_name)
dirs_mach = get_info(mach_dir, keys, out_dir_name, out_file_name)

comm.Barrier()
if comm.rank != 0:
    import sys
    print('PROCESS {} FINISHED'.format(comm.rank))
    sys.exit()

keys.append('Ma_ad_post')
keys.append('density_contrasts')
###########  Read in all means, stdevs of keys  ##########
def read_data(dirs):
    import h5py
    for i, dir in enumerate(dirs):
        try:
#        if True:
            f = h5py.File(dir[0]+out_dir_name+out_file_name+'.h5', 'r')
            Lz = np.exp(dir[4]/(1.5-dir[1]))-1
            z_basis = de.Chebyshev('z', len(f['z']), interval=[0, Lz],dealias=3/2)
            domain= de.Domain([z_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)
            current_field = domain.new_field()
            output_field  = domain.new_field()
            storage = dict()
            for key in keys:
                if key == 'density_contrasts':
                    storage[key] = (f[key][0], 0)
                elif key == 'Ma_ad':
                    storage[key] = (f[key+'_mean'][0]*np.sqrt(5/3), np.sqrt(5/3)*f[key+'_stdev'][0])
                elif key == 'Re_rms':
                    current_field['g'] = f[key]
    #                current_field.antidifferentiate('z', ('left', 0), out=output_field)
                    storage[key] = (current_field.interpolate(z=Lz/2)['g'][0], 0)
                else:
                    storage[key] = (f[key+'_mean'][0], f[key+'_stdev'][0])
            dirs[i][-1] = storage
        except:
            print("PROBLEMS READING OUTPUT FILE IN {:s}".format(dir[0]))
    return dirs

dirs_root = read_data(dirs_root)
dirs_mach = read_data(dirs_mach)

def plot_parameter_space_comparison(ax, x_key, y_key, data, grouping='eps', color='blue'):
    if grouping == 'eps':
        groups = np.unique(np.array(data)[:,1])
    if grouping == 'Ra':
        groups = np.unique(np.array(data)[:,2])

    if x_key == 'eps':
        x_key = 1
    if x_key == 'Ra':
        x_key = 2
    if x_key == 'Pr':
        x_key = 3
    if x_key == 'nrhocz':
        x_key = 4

    fig = plt.figure()

    for i, group in enumerate(groups):
        x_vals = []
        x_err  = []
        y_vals = []
        y_err  = []
        for dir in data:
            if grouping == 'eps' and group != dir[1] or dir[-1] == {}:
                continue
            if grouping == 'Ra' and group != dir[2] or dir[-1] == {}:
                continue
            if type(x_key) == int:
                x_vals.append(dir[x_key])
                x_err.append(0)
            else:
                x_vals.append(dir[-1][x_key][0])
                x_err.append(dir[-1][x_key][1])

            y_vals.append(dir[-1][y_key][0])
            y_err.append(dir[-1][y_key][1])
        if len(x_vals) == 0:
            continue
        x_vals, x_err, y_vals, y_err = zip(*sorted(zip(x_vals, x_err, y_vals, y_err)))
        print(group, x_vals, y_vals)
#        ax.plot(x_vals, y_vals, 'o', color=COLORS[i], marker='o')
        if grouping == 'eps':
            label = '$\epsilon = '
            pow = np.floor(np.log10(group))
            front = group/10**(pow)
            label += '{:1.1f} \\times 10^'.format(front)
            label += '{'
            label += '{:1d}'.format(int(pow))
            label += '}$'
            label = r'{:s}'.format(label)
        elif grouping == 'Ra':
            label = '$\mathrm{Ra} = '
            pow = np.floor(np.log10(group))
            front = group/10**(pow)
            label += '{:1.1f} \\times 10^'.format(front)
            label += '{'
            label += '{:1d}'.format(int(pow))
            label += '}$'
            label = r'{:s}'.format(label)
        else:
            label=''
        ax.errorbar(x_vals, y_vals, xerr=x_err, yerr=y_err, label=label, color=COLORS[i], marker=MARKERS[i], ls='None')
    return ax





        
fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.grid(which='major')
ax1 = plot_parameter_space_comparison(ax1, 'Ra', 'Nusselt_6', dirs_root, grouping='eps')
#ax1.set_xlabel('Ra')
ax1.set_ylabel(r'$\mathrm{Nu}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.legend(loc='lower right', fontsize=8)

ax2.grid(which='major')
ax2 = plot_parameter_space_comparison(ax2, 'Ra', 'Re_rms', dirs_root, grouping='eps')
ax2.set_ylim(1, 1e4)
ax2.set_xlabel(r'$\mathrm{Ra}$')
ax2.set_ylabel(r'$\mathrm{Re}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(loc='lower right', fontsize=8)

fig.savefig('./figs/re_and_nu_v_Ra.png', dpi=1200, bbox_inches='tight')

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_yticklabels():
    label.set_fontproperties('serif')


fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.grid(which='major')
ax1 = plot_parameter_space_comparison(ax1, 'Ra', 'Ma_ad_post', dirs_root, grouping='eps')
ax1.set_xlabel(r'$\mathrm{Ra}$')
ax1.set_ylabel(r'$\mathrm{Ma}_{\mathrm{ad}}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='lower right', fontsize=8)

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_yticklabels():
    label.set_fontproperties('serif')

ax2.grid(which='majors')
ax2 = plot_parameter_space_comparison(ax2, 'eps', 'Ma_ad_post', dirs_mach, grouping='Ra')
ax2.set_xlabel(r'$\epsilon$')
ax2.set_ylabel(r'$\mathrm{Ma}_{\mathrm{ad}}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(loc='lower right', fontsize=8)

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_yticklabels():
    label.set_fontproperties('serif')

fig.savefig('./figs/ma_v_Ra.png', dpi=1200, bbox_inches='tight')

fig = plt.figure(figsize=(FIGWIDTH, FIGWIDTH/golden_ratio))
ax1 = fig.add_subplot(1,1,1)

ax1.grid(which='major')
ax1 = plot_parameter_space_comparison(ax1, 'Ra', 'density_contrasts', dirs_root, grouping='eps')
ax1.set_xlabel(r'$\mathrm{Ra}$')
ax1.set_ylabel(r'$\rho(0)/\rho(L_z)$')
ax1.set_xscale('log')
ax1.legend(loc='center right', fontsize=8)

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')

fig.savefig('./figs/density_v_ra.png', dpi=1200, bbox_inches='tight')

#for dir in dirs_root:
#    print('ra: {}, eps: {}'.format(dir[2], dir[1]))
#    for key in dir[-1].keys():
#        print(key, dir[-1][key])
#    
