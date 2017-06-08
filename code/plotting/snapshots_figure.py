import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 11, 'axes.labelsize': 10})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.colorbar as colorbar
from base.plot_buddy import MovieBuddy
import numpy as np

FIGWIDTH=3+7.0/16
FIGHEIGHT= FIGWIDTH*1.25

start_file=59
n_files = 3

dir1="/nobackup/eanders/sp2017/fc_poly_hydro/FC_poly_fixed_constMu_constKappa_2D_nrhocz3_Ra1.00e6_Pr1_eps1e-4_a4_nusselt_fixedT"
dir2="/nobackup/eanders/sp2017/fc_poly_hydro/FC_poly_fixed_constMu_constKappa_2D_nrhocz3_Ra1.00e6_Pr1_eps5e-1_a4_nusselt_fixedT"
dir3="/nobackup/eanders/sp2017/fc_poly_hydro/FC_poly_fixed_constMu_constKappa_2D_nrhocz3_Ra4.64e7_Pr1_eps5e-1_a4_nusselt_fixedT_2"


stds=1

def add_snapshot_subplot(ax, dir, start_file,  do_cbar=False, figure_index=0, n_files=3):
    plotter = MovieBuddy(dir, max_files=n_files, start_file=start_file)
    plotter.add_subplot("s", 0, 0, zlabel="s'", sub_t_avg=True)
    plotter.analyze_subplots()
    slices = plotter.grab_whole_profile(plotter.local_files['slices'], plotter.local_writes_per_file,
                    subkey=['tasks'], profile_name=['s'])
    #max = plotter.ax[0]['mean']+stds*plotter.ax[0]['stdev']/2
    #min = plotter.ax[0]['mean']-stds*plotter.ax[0]['stdev']/2
    max = plotter.ax[0]['max_val']/3
    min = -plotter.ax[0]['max_val']/3
    img = ax.pcolormesh(plotter.xs, plotter.zs, slices['s'][figure_index,:,:]-plotter.ax[0]['t_avg'], cmap='RdBu_r',
           vmin=min, vmax=max)
    ax.set_xlim(np.min(plotter.xs), np.max(plotter.xs))
    ax.set_ylim(np.min(plotter.zs), np.max(plotter.zs))


    xticks = np.array([0, np.max(plotter.xs)/2, np.max(plotter.xs)])
    xticklabels = [r'${:1.1f}$'.format(tick) for tick in xticks]
    xticklabels[0] = r'${:1d}$'.format(0)
    plt.xticks(xticks, xticklabels, fontsize=11)
    yticks = np.array([0, np.max(plotter.zs)])
    yticklabels = [r'${:1.1f}$'.format(tick) for tick in yticks]
    yticklabels[0] = r'${:1d}$'.format(0)
    plt.yticks(yticks, yticklabels, fontsize=11)


    small_eps, small_ra = False, False
    plot_label = '$\epsilon='
    if plotter.atmosphere['epsilon'] < 0.1:
        plot_label += '10^{'
        plot_label += '{:1.0f}'.format(np.log10(plotter.atmosphere['epsilon']))
        plot_label += '}$'
    else:
        plot_label += '{:1.1f}$'.format(plotter.atmosphere['epsilon'])
        small_eps = True

    ra_log = np.log10(plotter.atmosphere['rayleigh'])
    plot_label += ' | $\mathrm{Ra} = 10^{'
    if np.floor(ra_log) == ra_log:
        plot_label += '{:1.0f}'.format(ra_log)
        small_ra = True
    else:
        plot_label += '{:1.2f}'.format(ra_log)
    plot_label += '}$'
    plot_label = r'({:s})'.format(plot_label)
#    plt.annotate(plot_label, coords, size=9, color='white', path_effects=[PathEffects.withStroke(linewidth=1.2, foreground='black')])


    if max > 0.1:
        cbar_label = '$\pm {:1.2f}$'.format(max)
    else:
        str = '{:1.2e}'.format(max)
        print(str)
        if 'e+0' in str:
            newstr = str.replace('e+0', '\\times 10^{')
        elif 'e-0' in str:
            newstr = str.replace('e-0', '\\times 10^{-')
        else:
            newstr = str.replace('e', '\\times 10^{')
        newstr += '}'
        cbar_label = '$\pm {:s}$'.format(newstr)

#    cbar_label += '  ({:s})'.format(plot_label)


    if do_cbar:
        cax, kw = colorbar.make_axes(ax, fraction=0.15, pad=0.03, aspect=5, anchor=(0,0), location='top')
        cbar = colorbar.colorbar_factory(cax, img, **kw)
        for label in cax.xaxis.get_ticklabels():
            label.set_visible(False)
        cax.tick_params(axis=u'both', which=u'both',length=0)

        trans = cax.get_yaxis_transform()
        cax.annotate(r'{:s}'.format(cbar_label), (1.02,0.01), size=11, color='black', xycoords=trans)
        cax.annotate(r'{:s}'.format(plot_label), (2.20,0.01), size=8, color='dimgrey', xycoords=trans)
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='15%', pad=0.01)
        cax.set_frame_on(False)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        for xlabel in cax.xaxis.get_ticklabels():
            xlabel.set_visible(False)
        for ylabel in cax.yaxis.get_ticklabels():
            ylabel.set_visible(False)
        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ann = ax.annotate(cbar_label, xy=(0, 1.03 ), size=11, color='black', xycoords=trans)
        if np.floor(ra_log) != ra_log :
            ann = ax.annotate(plot_label, xy=(xticks[1]*1.22, 1.03 ), size=8, color='dimgrey', xycoords=trans)
        else:
            ann = ax.annotate(plot_label, xy=(xticks[1]*1.30, 1.03 ), size=8, color='dimgrey', xycoords=trans)
#        plt.annotate(cbar_label, (0, yticks[-1]*1.01), size=8, color='black')


fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT), dpi=1200)
ax1 = fig.add_subplot(3,1,1)
add_snapshot_subplot(ax1, dir1, start_file, do_cbar=True, figure_index=20, n_files=3)
ax2 = fig.add_subplot(3,1,2)
add_snapshot_subplot(ax2, dir2, start_file, figure_index=64, n_files=4)
ax3 = fig.add_subplot(3,1,3)
add_snapshot_subplot(ax3, dir3, start_file, figure_index=9, n_files=2)

#ax2.plot([-0.11, 1.055], [1.175, 1.175], transform=ax2.transAxes, clip_on=False, color='black', linewidth=0.35)
#ax3.plot([-0.11, 1.055], [1.175, 1.175], transform=ax3.transAxes, clip_on=False, color='black', linewidth=0.35)

labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$']
axes   = [ax1, ax2, ax3]
y_coords = [3, 8, 8]
for i, ax in enumerate(axes):
    label = labels[i]
    trans = ax.get_yaxis_transform() # y in data untis, x in axes fraction
    ann = ax.annotate(label, xy=(-0.1, y_coords[i]), size=11, color='black', xycoords=trans)


print('saving png')
plt.savefig('./figs/snapshots_fig.png', bbox_inches='tight', dpi=1200)

