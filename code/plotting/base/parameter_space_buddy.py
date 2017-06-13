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
from matplotlib.colors import ColorConverter
from profile_buddy import ProfileBuddy, COLORS
import os
import numpy as np
import glob
from docopt import docopt
from mpi4py import MPI
comm_world = MPI.COMM_WORLD


import dedalus.public as de

EPSILON_ORDER=[1e-7, 1e0, 1e-4, 5e-1]
FORCE_WRITE=False

COLORS=['orange', 'red', 'green', 'indigo']
MARKERS=['s', 'o', 'd', '*']
MARKERSIZE=[5,4,5,7]

MARKERS_2=['p', '^', '8']
COLORS_2=['peru', 'gold', 'teal']
MARKERSIZE_2=[5,5,5]




class ParameterSpaceBuddy():
    
    def __init__(self, top_dir, parameters=['eps', 'ra', 'nrhocz', 'pr', 'a'], out_dir_name='/parameter_profiles/', out_file_name='parameter.h5' ):
        self.top_dir = top_dir
        for i in range(len(parameters)):
            parameters[i] = parameters[i].lower()
        self.parameters = parameters
        self.out_dir_name = out_dir_name
        self.out_file_name = out_file_name


    def get_info(self):
        #Get all of the directories and get info on them.
        dirs = []


        for i, dir in enumerate(glob.glob('{:s}/FC*/'.format(self.top_dir))):
            tag = dir.split(self.top_dir)[-1]
            info = dict()
            current_info = [dir,]
            continue_loop = False
            for stem in tag.split('_'):
                for parameter in self.parameters:
                    if parameter in stem.lower():
                        info[parameter] = float(stem.lower().split(parameter)[-1])
            for parameter in self.parameters:
                try:
                    current_info.append(info[parameter])
                except:
                    current_info.append(None)
                    continue_loop = True
                    if comm_world.rank == 0:
                        print('Dir {} does not have parameter {} in string'.format(dir, parameter))
            if continue_loop:
                continue

            current_info.append(dict())
            dirs.append(current_info)

        self.dir_info = dirs
        return self.dir_info

    def calculate_parameters(self, keys, force_write=False, start_file=1, n_files=1000):
        important_dirs = []
        for i, dir in enumerate(self.dir_info):
            if os.path.exists('{:s}/{:s}'.format(dir[0], self.out_dir_name)) and not force_write:
                if comm_world.rank == 0:
                    print('skipping {:s}, already solved there'.format(dir[0]))
                continue
            else:
                important_dirs.append(dir)

        base_start_file = start_file
        base_n_files = n_files

        for i, dir in enumerate(important_dirs):
            if np.mod(i, comm_world.size) != comm_world.rank:
                continue
#            try:
            if True:
                comm_new = comm_world.Create(comm_world.Get_group().Incl(np.ones(1)*comm_world.rank))

                #This exception exists because my old data is crappy -- consider removing this point.
                start_files = base_start_file
                n_files     = base_n_files
                if dir[1] == 1e-4 and dir[2] == 6.81e5:
                    start_file=175
                
                plotter = ProfileBuddy(dir[0], max_files=n_files, start_file=start_file, \
                                        write_cadence=1e10, file_dirs=['scalar', 'profiles'], comm=comm_new,
                                        outdir=self.out_dir_name)
                for j, key in enumerate(keys):
                    plotter.add_subplot(key, 0, j)
                plotter.analyze_subplots()
                plotter.communicate_profiles()
                plotter.save_profiles(filename=self.out_file_name.split('.h5')[0])
                plotter.make_plots(figsize=(3*len(keys), 8))
#            except:
#                print('AN ERROR HAS OCCURED IN {:s}'.format(dir[0]))
#                import sys
#                sys.stdout.flush()

    def read_files(self, keys):
        import h5py
        for i, dir in enumerate(self.dir_info):
#            try:
            if True:
                f = h5py.File(dir[0]+self.out_dir_name+self.out_file_name, 'r')

                Lz = np.exp(dir[self.parameters.index('nrhocz')+1]/(1.5-dir[self.parameters.index('eps')+1]))-1
                z_basis = de.Chebyshev('z', len(f['z']), interval=[0, Lz],dealias=3/2)
                domain= de.Domain([z_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)
                current_field = domain.new_field()
                output_field  = domain.new_field()
                storage = dict()
                for key_info in keys:
                    key, type = key_info
                    if type == 'val':
                        storage[key] = (np.log(f[key][0]), 0, 0)
                    elif type == 'midplane':
                        current_field['g'] = f[key]
                        storage[key] = (current_field.interpolate(z=Lz/2)['g'][0], 0, 0)
                    elif type == 'minmax':
                        storage[key] = (f[key+'_mean'][0], f[key+'_mean'][0] - np.min(f[key]), np.max(f[key]) - f[key+'_mean'][0])
                    elif type == 'logmaxovermin':
                        storage[key] = (np.log(np.max(f[key])/np.min(f[key])), 0, 0)
                    if 'Ma_ad' in key: #This is a workaround for my current incorrect implementation of Ma_ad (off by sqrt(gamma))
                        storage[key] = list(storage[key])
                        for i in range(len(storage[key])):
                            storage[key][i] = storage[key][i]*np.sqrt(5/3)
                for key in storage.keys():
                    if np.isnan(storage[key][0]):
                        raise
                dir[-1] = storage
#            except:
#                print("PROBLEMS READING OUTPUT FILE IN {:s}".format(dir[0]))
#                import sys
#                sys.stdout.flush()
        return self.dir_info


    def plot_parameter_space_comparison(self, ax, x_key, y_key, grouping='eps', color='blue', plot_log_x=False, plot_log_y=False,
            empty_markers=False, markersize=None, label=True, annotate='', onsets_norm=False, onsets=[], saturation=1):
        data = self.dir_info

        index = self.parameters.index(grouping.lower())+1
        if grouping == 'eps':
            groups = np.unique(np.array(data)[:,index])
            groups = EPSILON_ORDER
            colors = COLORS
            markers = MARKERS
            if markersize == None:
                markersize = MARKERSIZE
        elif grouping.lower() == 'ra':
            groups = np.unique(np.array(data)[:,index])
            colors = COLORS_2
            markers = MARKERS_2
            if markersize == None:
                markersize = MARKERSIZE_2
        else:
            colors = COLORS
            markers = MARKERS
            if markersize == None:
                markersize = MARKERSIZE
        try:
            x_key = int(self.parameters.index(x_key.lower()))+1
        except:
            print('x_key is a field, not a run parameter')

        fig = plt.figure()
        plot_points = dict()


        for i, group in enumerate(groups):
            x_vals = []
            x_err  = []
            y_vals = []
            y_err  = []
            for dir in data:
                if group != dir[index] or dir[-1] == {}:
                    continue
                try:
                    if onsets_norm:
                        x_vals.append(dir[x_key]/onsets[i])
                    else:
                        x_vals.append(dir[x_key])
                    x_err.append(0)
                except:
                    if onsets_norm:
                        x_vals.append(dir[-1][x_key][0]/onsets[i])
                        x_err.append((dir[-1][x_key][1]/onsets[i], dir[-1][x_key][2]/onsets[i]))
                    else:
                        x_vals.append(dir[-1][x_key][0])
                        x_err.append((dir[-1][x_key][1], dir[-1][x_key][2]))

                y_vals.append(dir[-1][y_key][0])
                y_err.append((dir[-1][y_key][1], dir[-1][y_key][2]))
            if len(x_vals) == 0:
                continue
            x_vals, x_err, y_vals, y_err = zip(*sorted(zip(x_vals, x_err, y_vals, y_err)))


            y_err_prev = y_err
            y_err = np.zeros((2, len(y_err)))
            y_err_almost = np.array(y_err_prev)
            for e in range(y_err_almost.shape[1]):
                y_err[e,:] = y_err_almost[:,e]
            x_err_prev = x_err
            x_err = np.zeros((2, len(x_err)))
            x_err_almost = np.array(x_err_prev)
            try:
                for e in range(x_err_almost.shape[1]):
                    x_err[e,:] = x_err_almost[:,e]
            except:
                for e in range(x_err_almost.shape[0]):
                    x_err[0,e] = x_err_almost[e]

    #        ax.plot(x_vals, y_vals, 'o', color=COLORS[i], marker='o')
            if grouping == 'eps':
                label = '$\epsilon = '
                pow = np.floor(np.log10(group))
                front = group/10**(pow)
                if pow >= -1:
                    label += '{:1.1f}$'.format(group)
                else:
                    if '{:1.1f}'.format(front) == '1.0':
                        label += '10^'
                    else:
                        label += '{:1.1f} \\times 10^'.format(front)
                    label += '{'
                    label += '{:1d}'.format(int(pow))
                    label += '}$'
                label = r'{:s}'.format(label)
            elif grouping.lower() == 'ra':
                label = '$\mathrm{Ra} = '
                pow = np.floor(np.log10(group))
                front = group/10**(pow)
                if '{:1.1f}'.format(front) == '1.0':
                    label += '10^'
                else:
                    label += '{:1.1f} \\times 10^'.format(front)
                label += '{'
                label += '{:1d}'.format(int(pow))
                label += '}$'
                label = r'{:s}'.format(label)
            else:
                label=''
            if plot_log_x:
                xvals, x_err = np.log10(xvals), np.log10(x_err)
            if plot_log_y:
                yvals, y_err = np.log10(yvals), np.log10(y_err)
            kwargs = dict()
            converter = ColorConverter()
            color = converter.to_rgba(colors[i], alpha=saturation)
            if empty_markers:
                kwargs['markerfacecolor'] = 'None'
                kwargs['markeredgecolor'] = color
                if annotate != '':
                    kwargs['markeredgewidth'] = 2
                else:
                    kwargs['markeredgewidth'] = 1
            if label:
                kwargs['label'] = label
            ax.errorbar(x_vals, y_vals, xerr=x_err, yerr=y_err, color=color, marker=markers[i], ms=markersize[i], ls='None', capsize=0, **kwargs)
            if annotate != '':
                for x, y in zip(x_vals, y_vals):
                    if empty_markers:
                        ax.plot(x, y, color=color, marker=markers[i], ms=markersize[i], \
                            markerfacecolor='None', markeredgecolor=color)
                        ax.plot(x, y, color=color, marker='o', ms=2)
                    else:
                        ax.plot(x, y, color=color, marker=markers[i], ms=markersize[i])
                        white = converter.to_rgba('white', alpha=0.8)
                        ax.plot(x, y, color=white, marker='o', ms=2)
                ax.plot(x_vals, y_vals, color=color, ls=':')
                        
                    
#                    ax.annotate(annotate, xy=(x,y), fontsize=5, color='white', verticalalignment='center', horizontalalignment='center')

            plot_points[group] = (x_vals, y_vals, x_err, y_err)
        return ax, plot_points, groups


