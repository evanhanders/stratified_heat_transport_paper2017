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
plt.style.use('classic')
from base.plot_buddy import ParameterSpaceBuddy, ProfileBuddy, COLORS
import os
import numpy as np
import glob
from docopt import docopt

import dedalus.public as de

EPSILON_ORDER=[1e-7, 1e0, 1e-4, 5e-1]
RA_ONSETS=[10.97, 11.15, 10.97, 10.06] 
threeD_onset_shift=1.0
twoD_saturation=1
threeD_saturation=0.35
legend_fontsize=9
FORCE_WRITE=False

def line_leastsq(x, y, y_uncert):
    x = np.copy(x)
    min_x = np.min(x)
    max_x = np.max(x)
    x_sub = 0
    if min_x != 0:
        if y_uncert[0] > y_uncert[-1]:
            x_sub = max_x
        else:
            x_sub = min_x
        x -= x_sub
    s = np.sum(1/y_uncert**2)#/max_y_u**2
    sx = np.sum(x / y_uncert**2)#/max_y_u**2
    sy = np.sum(y / y_uncert**2)#/max_y_u
    sx2 = np.sum(x**2 / y_uncert**2)#/max_y_u**2
    sxy = np.sum(x*y / y_uncert**2)#/max_y_u
    
    Delta = (s*sx2 - sx**2) 
    slope = (s * sxy - sx * sy) / Delta 
    intercept = (sx2*sy - sx*sxy) / Delta - x_sub*slope
    p = np.array((slope, intercept))
    covar = np.array(((s, -sx),(-sx, sx2)))/Delta
    return p, covar
START_FILE=100
N_FILES=1000
golden_ratio=1.618
FIGWIDTH=3+7.0/16
FIGHEIGHT=(FIGWIDTH/golden_ratio)*2.25 #2 plots, with padding for x-label.

THREED_MARKERSIZE = [8,6,6,10]


from mpi4py import MPI
comm = MPI.COMM_WORLD


#root_dir='/nobackup/eanders/sp2017/comps_data/'
root_dir='/nobackup/eanders/sp2017/fc_poly_hydro/'
mach_dir='/nobackup/eanders/sp2017/mach_eps/'
threeD_dir = '/nobackup/bpbrown/polytrope-evan-3D/'
mach_threeD_dir = '/nobackup/eanders/sp2017/mach_eps_3D/'
threeD_out_dir='/parameter_calcs/'
param_keys = ['Ra', 'eps', 'nrhocz', 'Pr']

keys = ['Nusselt', 'Nusselt_5', 'Nusselt_6', 'norm_5_kappa_flux_z',  
    'Re_rms', 'Ma_ad', 'Ma_iso', 'ln_rho1', 'T1', 'vel_rms',\
    'rho_full', 'T_full', 'kappa_flux_z', 'enthalpy_flux_z', 'PE_flux_z', 'KE_flux_z', 'viscous_flux_z', 'kappa_flux_z', 'kappa_flux_fluc_z']
plot_keys = [('Nusselt_6', 'minmax'), ('Re_rms', 'midplane'), ('Ma_ad_post', 'minmax'), ('density_contrasts', 'val'), ('rho_full', 'logmaxovermin')]

root_buddy = ParameterSpaceBuddy(root_dir, parameters=param_keys)
mach_buddy = ParameterSpaceBuddy(mach_dir, parameters=param_keys)
threeD_buddy = ParameterSpaceBuddy(threeD_dir, parameters=param_keys, out_dir_name=threeD_out_dir)
mach_threeD_buddy = ParameterSpaceBuddy(mach_threeD_dir, parameters=param_keys, out_dir_name=threeD_out_dir)

root_buddy.get_info()
mach_buddy.get_info()
threeD_buddy.get_info()
mach_threeD_buddy.get_info()

from docopt import docopt
args = docopt(__doc__)
if args['--calculate']:
    root_buddy.calculate_parameters(keys, force_write=FORCE_WRITE, start_file=START_FILE, n_files=N_FILES)
    mach_buddy.calculate_parameters(keys, force_write=FORCE_WRITE, start_file=START_FILE, n_files=N_FILES)
    threeD_buddy.calculate_parameters(keys, force_write=FORCE_WRITE, start_file=START_FILE, n_files=N_FILES)



comm.Barrier()
if comm.rank != 0:
    import sys
    print('PROCESS {} FINISHED'.format(comm.rank))
    sys.exit()


root_buddy.read_files(plot_keys)
mach_buddy.read_files(plot_keys)
threeD_buddy.read_files(plot_keys)
mach_threeD_buddy.read_files(plot_keys)

for dir in threeD_buddy.dir_info: #root_buddy.dir_info:
    print('ra: {}, eps: {}'.format(dir[2], dir[1]))
    for key in dir[-1].keys():
        print(key, dir[-1][key])
    print('  ')

dirs_root = root_buddy.dir_info
dirs_mach = mach_buddy.dir_info

#RE and NU plot 
fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.grid(which='major')
ax1, data, groups = root_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'Nusselt_6', grouping='eps',
            onsets_norm=True, onsets=RA_ONSETS, saturation=twoD_saturation)
ax1, data, groups = threeD_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'Nusselt_6', grouping='eps',
            markersize=THREED_MARKERSIZE, label=False, annotate='3', onsets_norm=True, onsets=RA_ONSETS,
            saturation=threeD_saturation)

x1s = np.logspace(2, 4, 100)
#x1s_long = np.logspace(3, 9, 100)
y1s = (10/7)*x1s**(1/3)
x2s = np.logspace(4, 7, 100)
y2s = 5*x2s**(1/5)

x_3ds = np.logspace(2, 6, 100)
y_3ds = 0.6*x_3ds**(2/7)
#y2s_diff = 1.0*x1s_long**(0.3)
ax1.plot(x1s/11, y1s, dashes=(5,1.5), color='black', label=r'$\mathrm{Ra}^{1/3}$')
ax1.plot(x2s/11, y2s, dashes=(2,1), color='black', label=r'$\mathrm{Ra}^{1/5}$')
ax1.plot(x_3ds/11, y_3ds, dashes=(5,1,2,1,2,1), color='dimgrey', label=r'$\mathrm{Ra}^{2/7}$')
ax1.annotate(r'$\mathrm{2D}$', xy=(5e2, 38), va='center', ha='center', fontsize=10)
ax1.annotate(r'$\mathrm{3D}$', xy=(3e3, 8), va='center', ha='center', fontsize=10, color='dimgrey')
#ax1.plot(x1s_long, y2s_diff, dashes=(2,1,2,1,5,1), color='goldenrod', label=r'$\mathrm{Ra}^{0.3}$')
#for key in data.keys():
#    x = np.array(data[key][0])
#    y = np.array(data[key][1])
##    y_err = np.array(data[key][-1])
##    print(y_err)
##    y_err = np.mean(y_err, axis=0)
#    y_err = np.ones_like(y)
#    try:
#        p, covar = line_leastsq(np.log10(x), np.log10(y), y_err)
#        ax1.plot(x, 10**(p[1])*(x)**(p[0]))
#        print(x, y)
#        print('full range nu', key, p)
#    except:
#        print('no high ra at {}'.format(key))

handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles[3:-2]] #remove error bars from label
vals, handles, labels = zip(*sorted(zip(1/np.array(groups), handles, labels[3:-2])))
legend1 = ax1.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, 
                                              borderpad=0.3, labelspacing=0.3, handletextpad=0.3, borderaxespad=0.3, fancybox=True)
#legend1 = ax1.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0)

#ax1.set_xlabel('Ra')
ax1.set_ylabel(r'$\mathrm{Nu}$')
ax1.set_ylim(1, 2e2)
ax1.set_xscale('log')
ax1.set_yscale('log')
handles, labels = ax1.get_legend_handles_labels()
handles = handles[:3]
labels = labels[:3]
ax1.legend(handles, labels, loc='upper left', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, 
                            labelspacing=0.1, handletextpad=0, fancybox=True, borderaxespad=0.3, columnspacing=0)
#ax1.legend(handles, labels, loc='upper left', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0.3)
ax1.add_artist(legend1)


ax2.grid(which='major')
ax2, data, groups = root_buddy.plot_parameter_space_comparison(ax2, 'Ra', 'Re_rms', grouping='eps',
            onsets_norm=True, onsets=RA_ONSETS, saturation=twoD_saturation)
ax2, data, groups = threeD_buddy.plot_parameter_space_comparison(ax2, 'Ra', 'Re_rms', grouping='eps',
            markersize=THREED_MARKERSIZE, label=False, annotate='3', onsets_norm=True, onsets=threeD_onset_shift*np.array(RA_ONSETS),
            saturation=threeD_saturation)
#for key in data.keys():
#    x = np.array(data[key][0])
#    y = np.array(data[key][1])
##    y_err = np.array(data[key][-1])
##    print(y_err)
##    y_err = np.mean(y_err, axis=0)
#    y_err = np.ones_like(y)
#    try:
#        p, covar = line_leastsq(np.log10(x), np.log10(y), y_err)
#        ax2.plot(x, 10**(p[1])*(x)**(p[0]))
#        print(x, y)
#        print('full range re', key, p)
#    except:
#        print('no high ra at {}'.format(key))

x1s = np.logspace(2, 8, 100)
y1s = x1s**(3/4)
x2s = np.logspace(4, 8, 100)
x2s_diff = np.logspace(4, 6, 100)
y2s = 10*x2s**(1/2)
y2s_diff = 2.2*x2s_diff**(2/3)

x_3ds = np.logspace(2,6,100)
y_3ds = 0.8*x_3ds**(1/2)
ax2.plot(x1s/11, y1s, dashes=(5,1.5), color='black', label=r'$\mathrm{Ra}^{3/4}$')
ax2.plot(x2s/11, y2s, dashes=(5,1,2,1,2,1), color='black', label=r'$\mathrm{Ra}^{1/2}$')
ax2.plot(x_3ds/11, y_3ds, dashes=(5,1,2,1,2,1), color='dimgrey')
ax2.annotate(r'$\mathrm{2D}$', xy=(5e2, 1500), va='center', ha='center', fontsize=10)
ax2.annotate(r'$\mathrm{3D}$', xy=(3e3, 60), va='center', ha='center', fontsize=10, color='dimgrey')
ax2.set_ylim(1, 5e4)
ax2.set_xlim(1e0, 1e7)
ax2.set_xlabel(r'$\mathrm{Ra}/\mathrm{Ra}_{\mathrm{crit}}$')
ax2.set_ylabel(r'$\mathrm{Re}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
handles, labels = ax2.get_legend_handles_labels()
handles = [h[0] for h in handles[2:]] #remove error bars from label
vals, handles, labels = zip(*sorted(zip(1/np.array(groups), handles, labels[2:])))
#legend1 = ax2.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0)
legend1 = ax2.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, 
                                              borderpad=0.3, labelspacing=0.3, handletextpad=0.3, borderaxespad=0.3, fancybox=True)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[:2], labels[:2], loc='upper left', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, 
                            labelspacing=0.1, handletextpad=0, fancybox=True, borderaxespad=0.3, columnspacing=0)
#ax2.legend(handles[:2], labels[:2], loc='upper left', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0.3)
ax2.add_artist(legend1)


for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_yticklabels():
    label.set_fontproperties('serif')

labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$']
axes   = [ax1, ax2]
y_coords = [1e2, 1e4]
for i, ax in enumerate(axes):
    label = labels[i]
    trans = ax.get_yaxis_transform() # y in data untis, x in axes fraction
    ann = ax.annotate(label, xy=(-0.21, y_coords[i]), size=11, color='black', xycoords=trans)




fig.savefig('./figs/re_and_nu_v_Ra.png', dpi=1200, bbox_inches='tight')



#MA plot
fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
ax2 = fig.add_subplot(2,1,1)
ax1 = fig.add_subplot(2,1,2)



ax1.grid(which='major')
ax1, data, groups = root_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'Ma_ad_post', grouping='eps',
            onsets_norm=True, onsets=RA_ONSETS, saturation=twoD_saturation)
ax1, data, groups = threeD_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'Ma_ad_post', grouping='eps',
            markersize=THREED_MARKERSIZE, label=False, annotate='3', onsets_norm=True, onsets=threeD_onset_shift*np.array(RA_ONSETS),
            saturation=threeD_saturation)
#for key in data.keys():
#    x = np.array(data[key][0])
#    y = np.array(data[key][1])
#    y_err = np.array(data[key][-1])
#    print(y_err)
#    y_err = np.mean(y_err, axis=0)
#    x1 = x[np.where(np.logical_and(x>=2.15e1, x<=1e4))]
#    y1 = y[np.where(np.logical_and(x>=2.15e1, x<=1e4))]
#    y_err_1 = np.ones(len(y1))#y_err[np.where(np.logical_and(x>=2.15e1, x<=1e4))]
#    x2 = x[np.where(x>1e4)]
#    y2 = y[np.where(x>1e4)]
#    y_err_2 = np.ones(len(y2))
#    print(y_err_1)
#    try:
#        p, covar = line_leastsq(np.log10(x1), np.log10(y1), y_err_1)
#        ax1.plot(x1, 10**(p[1])*(x1)**(p[0]))
#        print(key, p)
#        p, covar = line_leastsq(np.log10(x2), np.log10(y2), y_err_2)
#        ax1.plot(x2, 10**(p[1])*(x2)**(p[0]))
#        print(key, p)
#    except:
#        print('no high ra at {}'.format(key))
x1s = np.logspace(1.5, 4, 100)
y1s = 0.03*x1s**(1/4)
x1s_long = np.logspace(1.5, 6, 100)
y1s_long = 0.03*x1s_long**(1/4)/60
x2s = np.logspace(4, 7, 100)
y2s = 0.5*x2s**(-1/20)
y3s = 2e-3*x2s**(1/10)
ax1.plot(x1s/11, y1s, dashes=(5,1.5), color='black', label=r'$\mathrm{Ra}^{1/4}$')#, label=r'$\mathrm{Ra}^{1/3}$')
ax1.plot(x1s_long/11, y1s_long, dashes=(5,1.5), color='black')#, label=r'$\mathrm{Ra}^{1/3}$')
ax1.plot(x1s/11, y1s/3.16e3, dashes=(5,1.5), color='black')#, label=r'$\mathrm{Ra}^{1/3}$')
#ax1.plot(x2s, y2s, dashes=(2,1), color='black')#, label=r'$\mathrm{Ra}^{1/5}$')
#ax1.plot(x2s, y3s, dashes=(2,1,5,1), color='black')#, label=r'$\mathrm{Ra}^{1/5}$')

ax1.set_xlabel(r'$\mathrm{Ra}/\mathrm{Ra}_{\mathrm{crit}}$')
ax1.set_ylabel(r'$\mathrm{Ma}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1e0, 1e7)
handles, labels = ax1.get_legend_handles_labels()
legend1 = ax1.legend(handles[:1], labels[:1], loc='center right', fontsize=legend_fontsize, numpoints=1, 
                                              borderpad=0.3, labelspacing=0.3, handletextpad=0.3, borderaxespad=0.3, fancybox=True)

handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles[1:]] #remove error bars from labels
vals, handles, labels = zip(*sorted(zip(1/np.array(groups), handles, labels[1:])))
ax1.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, borderpad=0.2, 
                            labelspacing=0.1, handletextpad=0, fancybox=True, borderaxespad=0.3, columnspacing=0)
ax1.add_artist(legend1)

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_yticklabels():
    label.set_fontproperties('serif')

ax2.grid(which='major')
ax2, data, groups = mach_threeD_buddy.plot_parameter_space_comparison(ax2, 'eps', 'Ma_ad_post', grouping='Ra',
            markersize=np.array(THREED_MARKERSIZE)*1.3, label=False, annotate='3',
            saturation=threeD_saturation)
ax2, data, groups = mach_buddy.plot_parameter_space_comparison(ax2, 'eps', 'Ma_ad_post', grouping='Ra')

#for key in data.keys():
#    x = np.array(data[key][0])
#    y = np.array(data[key][1])
#    y_err = np.array(data[key][-1])
#    y_err = np.mean(y_err, axis=0)
#    p, covar = line_leastsq(np.log10(x), np.log10(y), y_err)
#    ax2.plot(x, 10**(p[1])*(x)**(p[0]))
#    print(key, p)
x1s = np.logspace(-7, 1, 100)
y1s = (0.5)*x1s**(1/2)
ax2.plot(x1s, y1s, dashes=(5,1.5), color='black', label=r'$\epsilon^{1/2}$')

handles, labels = ax2.get_legend_handles_labels()
legend1 = ax2.legend(handles[:1], labels[:1], loc='upper left', fontsize=legend_fontsize, numpoints=1, 
                                              borderpad=0.3, labelspacing=0.3, handletextpad=0.3, borderaxespad=0.3, fancybox=True)
#legend1 = ax2.legend(handles[:1], labels[:1], loc='upper left', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0.3)

ax2.set_xlabel(r'$\epsilon$', labelpad=-3)
ax2.set_ylabel(r'$\mathrm{Ma}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
handles, labels = ax2.get_legend_handles_labels()
handles = [h[0] for h in handles[4:]] #remove error bars from labels
vals, handles, labels = zip(*sorted(zip(1/np.array(groups), handles, labels[4:])))
ax2.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, 
                            labelspacing=0.1, handletextpad=0, fancybox=True, borderaxespad=0.3, columnspacing=0)
#ax2.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0)
ax2.add_artist(legend1)
ax2.set_xlim(7e-8, 2e0)

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax2.get_yticklabels():
    label.set_fontproperties('serif')



labels = [r'$\mathrm{(b)}$', r'$\mathrm{(a)}$']
axes   = [ax1, ax2]
y_coords = [3.16e0, 3.16e0]
for i, ax in enumerate(axes):
    label = labels[i]
    trans = ax.get_yaxis_transform() # y in data untis, x in axes fraction
    ann = ax.annotate(label, xy=(-0.21, y_coords[i]), size=11, color='black', xycoords=trans)

fig.savefig('./figs/ma_v_Ra.png', dpi=1200, bbox_inches='tight')


#Density plot

fig = plt.figure(figsize=(FIGWIDTH, FIGWIDTH/golden_ratio))
ax1 = fig.add_subplot(1,1,1)

ax1.grid(which='major')
ax1, data, groups3 = threeD_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'rho_full', grouping='eps',
            markersize=[10, 7, 7, 13], label=False, annotate='3', empty_markers=True, onsets_norm=True, onsets=threeD_onset_shift*np.array(RA_ONSETS),
            saturation=threeD_saturation)
ax1, data, groups3 = threeD_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'density_contrasts', grouping='eps',
            saturation=threeD_saturation,
            markersize=[10, 7, 7, 13], label=False, annotate='3', onsets_norm=True, onsets=threeD_onset_shift*np.array(RA_ONSETS))
ax1, data, groups = root_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'rho_full', grouping='eps', empty_markers=True,
                onsets_norm=True, onsets=RA_ONSETS, saturation=twoD_saturation)
ax1, data, groups = root_buddy.plot_parameter_space_comparison(ax1, 'Ra', 'density_contrasts', grouping='eps',
                onsets_norm=True, onsets=RA_ONSETS, saturation=twoD_saturation)
ax1.set_xlabel(r'$\mathrm{Ra}/\mathrm{Ra}_{\mathrm{crit}}$')
ax1.set_ylabel(r'$n_{\rho}$')
ax1.set_ylim(-1, 3.3)
ax1.set_xlim(1e0, 1e7)
ax1.set_xscale('log')
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles] #remove error bars from labels
vals, handles, labels = zip(*sorted(zip(1/np.array(groups), handles[-4:], labels[-4:])))
ax1.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, 
                            labelspacing=0.1, handletextpad=0, fancybox=True, borderaxespad=0.3, columnspacing=0)
#ax1.legend(handles, labels, loc='lower right', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0)

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')

fig.savefig('./figs/density_v_ra.png', dpi=1200, bbox_inches='tight')
 
fig = plt.figure(figsize=(FIGWIDTH, FIGWIDTH/golden_ratio))
ax1 = fig.add_subplot(1,1,1)

ax1.grid(which='major')
ax1, data, groups = root_buddy.plot_parameter_space_comparison(ax1, 'Ma_ad_post', 'rho_full', grouping='eps', empty_markers=True,
                onsets_norm=True, onsets=RA_ONSETS, saturation=twoD_saturation)
ax1, data, groups = threeD_buddy.plot_parameter_space_comparison(ax1, 'Ma_ad_post', 'rho_full', grouping='eps',
            markersize=THREED_MARKERSIZE, label=False, annotate='3', empty_markers=True, onsets_norm=True, onsets=RA_ONSETS, saturation=threeD_saturation)
ax1, data, groups = root_buddy.plot_parameter_space_comparison(ax1, 'Ma_ad_post', 'density_contrasts', grouping='eps',
                onsets_norm=True, onsets=RA_ONSETS, saturation=twoD_saturation)
ax1, data, groups3 = threeD_buddy.plot_parameter_space_comparison(ax1, 'Ma_ad_post', 'density_contrasts', grouping='eps',
            markersize=THREED_MARKERSIZE, label=False, annotate='3', onsets_norm=True, onsets=RA_ONSETS, saturation=threeD_saturation)
ax1.set_xlabel(r'$\mathrm{Ma}$')
ax1.set_ylabel(r'$n_{\rho}$')
ax1.set_xlim(1e0, 1e7)
ax1.set_ylim(-1, 3.3)
ax1.set_xscale('log')
handles, labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles] #remove error bars from labels
vals, handles, labels = zip(*sorted(zip(1/np.array(groups), handles[-7:-3], labels[-7:-3])))
ax1.legend(handles, labels, loc='lower left', fontsize=legend_fontsize, numpoints=1, borderpad=0.3, labelspacing=0.3, handletextpad=0)

for label in ax1.get_xticklabels():
    label.set_fontproperties('serif')
for label in ax1.get_yticklabels():
    label.set_fontproperties('serif')

fig.savefig('./figs/density_v_ma.png', dpi=1200, bbox_inches='tight')
 


