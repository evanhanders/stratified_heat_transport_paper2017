import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 11, 'axes.labelsize': 10})
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
plt.style.use('classic')
import matplotlib.patheffects as PathEffects
import matplotlib.colorbar as colorbar
from base.plot_buddy import MovieBuddy
import numpy as np

FIGWIDTH=3+7.0/16
FIGHEIGHT= FIGWIDTH*1

start_file=49
n_files = 3

dir1="/nobackup/eanders/sp2017/comps_data/FC_poly_fixed_constMu_constKappa_nrhocz3_Ra1.00e6_Pr1_eps1e-4_nusselt_fixedT_highres/"
#dir1="/nobackup/eanders/sp2017/fc_poly_hydro/FC_poly_fixed_constMu_constKappa_2D_nrhocz3_Ra1.00e6_Pr1_eps1e-4_a4_nusselt_fixedT"
dir2="/nobackup/eanders/sp2017/fc_poly_hydro/FC_poly_fixed_constMu_constKappa_2D_nrhocz3_Ra1.00e6_Pr1_eps5e-1_a4_nusselt_fixedT"
dir3="/nobackup/eanders/sp2017/fc_poly_hydro/FC_poly_fixed_constMu_constKappa_2D_nrhocz3_Ra4.64e7_Pr1_eps5e-1_a4_nusselt_fixedT"
dir4="/nobackup/bpbrown/polytrope-evan-3D/FC_poly_3D_nrhocz3_Ra1e6_Pr1_eps1e-4_a4/"


stds=1

def add_snapshot_subplot(ax, dir, start_file,  do_cbar=False, figure_index=0, n_files=3, field='s', horiz_slice=False, plot_label=None, dims='$\mathrm{2D}$ | '):
    plotter = MovieBuddy(dir, max_files=n_files, start_file=start_file)
    plotter.add_subplot(field, 0, 0, zlabel="s'", sub_t_avg=True)
    plotter.analyze_subplots()
    slices = plotter.grab_whole_profile(plotter.local_files['slices'], plotter.local_writes_per_file,
                    subkey=['tasks'], profile_name=[field])
    #max = plotter.ax[0]['mean']+stds*plotter.ax[0]['stdev']/2
    #min = plotter.ax[0]['mean']-stds*plotter.ax[0]['stdev']/2
    max = plotter.ax[0]['max_val']/3
    min = -plotter.ax[0]['max_val']/3

    if horiz_slice:
        xs, ys = plotter.y_xx, plotter.y_yy
    else:
        xs, ys = plotter.xs, plotter.zs
    if type(plotter.y) == type(None):
        img = ax.pcolormesh(xs, ys, slices[field][figure_index,:,:]-plotter.ax[0]['t_avg'], cmap='RdBu_r',
           vmin=min, vmax=max)
    else:
        if horiz_slice:
            img = ax.pcolormesh(xs, ys, slices[field][figure_index,:,:][:,:,0]-np.mean(plotter.ax[0]['t_avg']), cmap='RdBu_r',
               vmin=min, vmax=max)
        else:
            img = ax.pcolormesh(xs, ys, slices[field][figure_index,:,:][:,0,:]-plotter.ax[0]['t_avg'], cmap='RdBu_r',
               vmin=min, vmax=max)
    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))




    xticks = np.array([0, np.max(xs)/2, np.max(xs)])
    xticklabels = [r'${:1.1f}$'.format(tick) for tick in xticks]
    xticklabels[0] = r'${:1d}$'.format(0)
    plt.xticks(xticks, xticklabels, fontsize=8)
    yticks = np.array([0, np.max(ys)])
    yticklabels = [r'${:1.1f}$'.format(tick) for tick in yticks]
    yticklabels[0] = r'${:1d}$'.format(0)
    plt.yticks(yticks, yticklabels, fontsize=8)

    custom_label=True
    if type(plot_label) == type(None):
        custom_label=False
        small_eps, small_ra = False, False
        plot_label = '{:s}$\epsilon='.format(dims)
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
    else:
        plot_label = r'(${:s}$)'.format(plot_label)
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
        cax.annotate(r'{:s}'.format(cbar_label), (1.02,0.04), size=8, color='black', xycoords=trans)
        cax.annotate(r'{:s}'.format(plot_label), (2.08,0.04), size=8, color='dimgrey', xycoords=trans)
    else:
        divider = make_axes_locatable(ax)
        if horiz_slice:
            cax = divider.append_axes('top', size='40%', pad=0.01)
            cbx = divider.append_axes('bottom', size='30%', pad=0.01)
            cbx.set_frame_on(False)
            cbx.get_xaxis().set_visible(False)
            cbx.get_yaxis().set_visible(False)
            for xlabel in cbx.xaxis.get_ticklabels():
                xlabel.set_visible(False)
            for ylabel in cbx.yaxis.get_ticklabels():
                ylabel.set_visible(False)
        else:
            cax = divider.append_axes('top', size='10%', pad=0.06)
        cax.set_frame_on(False)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        for xlabel in cax.xaxis.get_ticklabels():
            xlabel.set_visible(False)
        for ylabel in cax.yaxis.get_ticklabels():
            ylabel.set_visible(False)
        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ann = ax.annotate(cbar_label, xy=(-0.05, 1.05 ), size=8, color='black', xycoords=trans)
        if not custom_label:
            if np.floor(ra_log) != ra_log:
                    ann = ax.annotate(plot_label, xy=(xticks[-1]*0.48, 1.05 ), size=8, color='dimgrey', xycoords=trans)
            elif small_eps:
                ann = ax.annotate(plot_label, xy=(xticks[-1]*0.53, 1.05 ), size=8, color='dimgrey', xycoords=trans)
            else:
                ann = ax.annotate(plot_label, xy=(xticks[-1]*0.49, 1.05 ), size=8, color='dimgrey', xycoords=trans)

        else:
                ann = ax.annotate(plot_label, xy=(xticks[1]*1.25, 1.05 ), size=8, color='dimgrey', xycoords=trans)
            
#        plt.annotate(cbar_label, (0, yticks[-1]*1.01), size=8, color='black')


fig = plt.figure(figsize=(2*FIGWIDTH, FIGHEIGHT), dpi=1200)
plt.subplots_adjust(wspace=0.4)
ax1 = plt.subplot2grid((3,4), (0,0), colspan=2)
add_snapshot_subplot(ax1, dir1, 85, do_cbar=True, figure_index=20, n_files=3)
ax2 = plt.subplot2grid((3,4), (1,0), colspan=2)
add_snapshot_subplot(ax2, dir2, 70, figure_index=64, n_files=4)
ax3 = plt.subplot2grid((3,4), (2,0), colspan=2)
add_snapshot_subplot(ax3, dir3, 70, figure_index=5, n_files=2)

#3D
ax4 = plt.subplot2grid((3,4), (0,2), colspan=2)
add_snapshot_subplot(ax4, dir4, 30, figure_index=9, n_files=2, dims= '$\mathrm{3D}$ | ')
ax5 = plt.subplot2grid((3,4), (1,2), rowspan=2)
add_snapshot_subplot(ax5, dir4, 30, figure_index=9, n_files=2, field='s midplane', horiz_slice=True, plot_label='z=L_z/2')
ax6 = plt.subplot2grid((3,4), (1,3), rowspan=2)
add_snapshot_subplot(ax6, dir4, 30, figure_index=9, n_files=2, field='s near top', horiz_slice=True, plot_label='z=L_z')

#ax2.plot([-0.11, 1.055], [1.175, 1.175], transform=ax2.transAxes, clip_on=False, color='black', linewidth=0.35)
#ax3.plot([-0.11, 1.055], [1.175, 1.175], transform=ax3.transAxes, clip_on=False, color='black', linewidth=0.35)

labels = [r'$\mathrm{(a)}$', r'$\mathrm{(b)}$', r'$\mathrm{(c)}$', r'$\mathrm{(d)}$', r'$\mathrm{(e)}$', r'$\mathrm{(f)}$']
axes   = [ax1, ax2, ax3, ax4, ax5, ax6]
x_coords = [-0.1, -0.1, -0.1, -0.1, -0.2, -0.2]
y_coords = [3, 8, 8, 3, 10.7, 10.7]
for i, ax in enumerate(axes):
    label = labels[i]
    trans = ax.get_yaxis_transform() # y in data untis, x in axes fraction
    ann = ax.annotate(label, xy=(x_coords[i], y_coords[i]), size=9, color='black', xycoords=trans)


print('saving png')
plt.savefig('./figs/snapshots_fig.png', bbox_inches='tight', dpi=600)

