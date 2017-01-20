from plot_buddy_base import *
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
from scipy.stats.mstats import mode
import matplotlib.colorbar as colorbar

POSITIVE_DEFINITE = ['enstrophy']

class MovieBuddy(PlotBuddy):
     
    def __init__(self, dir, profile_post_file=None, **kwargs):
        super(MovieBuddy, self).__init__(dir, **kwargs)
        if self.idle: return
        #Figure out the number range of files for this process.
        self.file_number_start = self.files_below + 1
        self.file_number_end = self.files_below +  self.n_files
        self.global_total_writes_below = np.sum(self.global_writes_per_file[:self.files_below])
        if profile_post_file != None:
            self.profile_post_file = profile_post_file
        else:
            self.profile_post_file = None

    def add_subplot(self, field, rindex, cindex, hlabel='x', vlabel='z', \
                        zlabel=None, ncols=1, nrows=1, cmap='RdYlBu_r',
                        sub_t_avg=False, bare=False):
        '''
            Adds a subplot to the list of subplots that the plotter will track.

            field     -- the field being plottedin this subplot
            rindex    -- row index of the subplot, starting from zero
            cindex    -- column index of the subplot, starting from zero
            hlabel    -- the label for the horizontal axis ('x', 't', or something else)
            vlabel    -- the label for the vertical axis ('z', or something else)
            zlabel    -- the label for the third axis (the colormap axis)
            colspan   -- The number of columns that the subplot spans across
            rowspan   -- The number of rows that the subplot spans across
            cmap      -- The colormap of the plot
            sub_t_avg -- If True, subtract the time average from each movie frame.
            bare      -- If bare, this is for a public talk type movie.
        '''
        if self.idle: return
        plot_info = dict()
        plot_info['field']    = field
        plot_info['position'] = (rindex, cindex)
        plot_info['colspan']  = ncols
        plot_info['rowspan']  = nrows
        plot_info['hlabel']   = hlabel
        plot_info['vlabel']   = vlabel
        plot_info['zlabel']   = zlabel
        plot_info['cmap']     = cmap
        plot_info['sub_t_avg']= sub_t_avg
        plot_info['bare']     = bare
        self.ax.append(plot_info)


    def plot_field(self, xs, zs, field, ax, field_name, cmap='RdYlBu_r', min=None, max=None,\
                    xlims=None, ylims=None, mod=None, plot_title=True, function='colormap',
                    bare=False):
        '''
            Plots a colormap of a given field.

            xs -- a 2D grid of x-values
            zs -- a 2D grid of z-values
            field -- a 2D grid of values in the x-z plane for another parameter.
            ax -- the Axis subplot on which to plot the field.
            field_name -- The physical name that the numbers in 'field' represent
            cmap -- the colormap of the plot
            min, max -- the min and max values of the colormap to plot
            xlims, ylims -- the boundaries on the x- and y- coordinates.
            mod -- A modification on the field to alter the movie.  Currently accepts:
                    "up" -- upflows
                    "down" -- downflows
                    "pos"  -- Positive parts only
                    "neg"  -- Negative parts only
            bare -- If bare, then don't plot up any axis information (public talks, etc.)

        '''
        if self.idle: return
        if max == None:
            max = np.max(field)
        if min == None:
            min = np.min(field)

        if function == 'colormap':
            plot = ax.pcolormesh(xs, zs, field, cmap=cmap, vmin=min, vmax=max)
            if not bare:
                xticks = np.array([1, np.max(xs)/2, np.max(xs)])
                yticks = np.array([1, np.max(zs)/2, np.max(zs)])
                plt.xticks(xticks, [r'${:1.2f}$'.format(tick) for tick in xticks], fontsize=11)
                plt.yticks(yticks, [r'${:1.2f}$'.format(tick) for tick in yticks], fontsize=11)

                if self.atmosphere['atmosphere_name'] == 'single polytrope':
                    plot_label = '$\epsilon='
                    if self.atmosphere['epsilon'] < 0.1:
                        plot_label += '10^{'
                        plot_label += '{:1.0f}'.format(np.log10(self.atmosphere['epsilon']))
                        plot_label += '}$'
                    else:
                        plot_label += '{:1.1f}$'.format(self.epsilon)
                        small_eps = True

                    ra_log = np.log10(self.atmosphere['rayleigh'])
                    plot_label += ' | $\mathrm{Ra} = 10^{'
                    if np.floor(ra_log) == ra_log:
                        plot_label += '{:1.0f}'.format(ra_log)
                        small_ra = True
                    else:
                        plot_label += '{:1.2f}'.format(ra_log)
                    plot_label += '}$'
                    plot_label = r'{:s}'.format(plot_label)
                else:
                    print('ERROR: Unknown atmosphere type')
                    plot_label=''

                if max > 0.1:
                    cbar_label = '$\pm {:1.2f}$'.format(max)
                else:
                    str = '{:1.2e}'.format(max)
                    if 'e+0' in str:
                        newstr = str.replace('e+0', '\\times 10^{')
                    elif 'e-0' in str:
                        newstr = str.replace('e-0', '\\times 10^{-')
                    else:
                        newstr = str.replace('e', '\\times 10^{')
                    newstr += '}'
                    if min != 0:
                        cbar_label = '$\pm {:s}$'.format(newstr)
                    else:
                        cbar_label = '$min: 0 max: {:s}$'.format(newstr)

                cbar_label += '  ({:s})'.format(plot_label)
                cbar_label += ' {:s}'.format(field_name)

                divider = make_axes_locatable(ax)
                cax, kw = colorbar.make_axes(ax, fraction=0.07, pad=0.03, aspect=5, anchor=(0,0), location='top')
                cbar = colorbar.colorbar_factory(cax, plot, **kw)
                trans = cax.get_yaxis_transform()
                cax.annotate(r'{:s}'.format(cbar_label), (1.02,0.01), size=11, color='black', xycoords=trans)
                cax.tick_params(axis=u'both', which=u'both',length=0)
                cax.set_xticklabels([])
                cax.set_xticks([])
#                for label in cax.xaxis.get_ticklabels():
#                    label.set_visible(False)
#                for label in cax.yaxis.get_ticklabels():
#                    label.set_visible(False)


        elif function == 'contour':
            field[np.where(field > max)] = max
            field[np.where(field < min)] = min
            plot = ax.contour(xs, zs, field, 7, cmap=plt.cm.gray)

        if xlims == None or len(xlims) != 2:
            ax.set_xlim(np.min(xs), np.max(xs))
        else:
            ax.set_xlim(xlims[0], xlims[1])
        if ylims == None or len(ylims) != 2:
            ax.set_ylim(np.min(zs), np.max(zs))
        else:
            ax.set_ylim(ylims[0], ylims[1])

        if bare:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            

    def get_time_avg(self, field):
        ''' Takes a 3D Time x Width x Height field array and returns the time average o
            the vertical profile

            TODO generalize this to 3D
        '''
        if self.idle: return

        local_counts  = np.zeros(1, np.int16)
        global_counts = np.zeros(1, np.int16)
        local_sum     = np.zeros(1, np.float64)
        global_sum    = np.zeros(1, np.float64)


        local_counts[0] = field.shape[0]

        collapsed_profile = np.mean(field, axis=1)
        collapsed_profile = np.sum(collapsed_profile, axis=0)
        local_sum[0] = collapsed_profile[0]

        self.comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)

        return global_sum/global_counts[0]

    def analyze_subplots(self):
        '''
            Takes provided information about subplots and gets relevant info on how
            to set up those subplots
        '''
        if self.idle: return

        field_names = []
        total_width = 0
        total_height = 0
        for i, ax in enumerate(self.ax):
            field_names.append(ax['field'])
            if ax['position'][0] + 1 > total_height:
                total_height = ax['position'][0] + 1
            if ax['position'][1] + 1 > total_width:
                total_width = ax['position'][1] + 1
        self.plot_dimensions = (total_height, total_width) #rows, ncols

        slices_fields = self.grab_whole_profile(self.local_files['slices'],\
                                                self.local_writes_per_file,
                                                subkey=['tasks']*len(field_names), \
                                                profile_name=field_names)
        for i, ax in enumerate(self.ax):
            ax['data'] = slices_fields[ax['field']]
            if ax['sub_t_avg']:
                ax['t_avg'] = self.get_time_avg(ax['data'])
                ax['data'] -= ax['t_avg']
                flattened = np.sort(np.abs(ax['data'].flatten()))

                ax['max_val'] = flattened[int(0.98*len(flattened))]

    def make_plots(self, figsize=None, outdir='snapshots', filename='snapshots', write_number_start=1,\
                            dpi=300, cbar_factor=0.2, length_div=1, plot_title=True, n_mode_memory=100):
        '''
            Create all of the plots!
        '''
        if self.idle: return


        output_directory = self.root_dir + '/' + outdir + '/'

        if figsize == None:
            figsize = (self.plot_dimensions[1]*(2*self.atmosphere['aspect_ratio']), self.plot_dimensions[0]*2+0.5)
        if self.cw_rank == 0 and not os.path.exists('{:s}'.format(output_directory)):
            os.mkdir('{:s}'.format(output_directory))
        logger.info('saving figures to {}'.format(output_directory))

        count = int(self.global_total_writes_below+write_number_start)
        writes = int(np.sum(self.local_total_writes))
        num_in_file = 0

        for i in range(writes):
            logger.info('Plotting {}/{}'.format(i+1,writes))
            movie_count = 0
            fig = plt.figure(figsize=figsize)
            current_time = self.local_times[i]
            axes = dict()
            #Plot each subplot
            for k, ax in enumerate(self.ax):
                position = ax['position']
                colspan  = ax['colspan']
                rowspan  = ax['rowspan']
                field    = ax['field']
                hlabel   = ax['hlabel']
                vlabel   = ax['vlabel']
                zlabel   = ax['zlabel']
                cmap     = ax['cmap']
                bare     = ax['bare']

                if position in axes.keys():
                    axis = axes[position]
                else:
                    axis = plt.subplot2grid(self.plot_dimensions, position, colspan=colspan, rowspan=rowspan)
                    axes[position] = axis

                field_base  = ax['data'][i,:]
                max         = ax['max_val']
                min         = -ax['max_val']
                if field in POSITIVE_DEFINITE:
                    min=0

                if zlabel == None:
                    zlabel = field
                
                self.plot_field(self.xs/length_div, self.zs/length_div, field_base, axis, \
                            zlabel, cmap=cmap, min=min, max=max, plot_title=plot_title,\
                            bare=bare)#, extra_label=field_base)
                if length_div != 1:
                    axis.set_xlabel('x / L', fontsize=11)
                    axis.set_ylabel('z / L', fontsize=11)
                else:
                    axis.set_xlabel('x', fontsize=11)
                    axis.set_ylabel('z', fontsize=11)
            
            if plot_title and not bare:                            
                title_string = 't = {:1.2e}'.format(current_time)
                try:
                    title_string += '; t/t_buoy = {:1.2e}'.format(current_time/self.atmosphere['t_buoy'])
                    title_string += '; t/t_therm = {:1.2e}'.format(current_time/self.atmosphere['t_therm'])
                except:
                    print("ERROR: Cannot find t_buoy or t_therm")
                fig.suptitle(title_string, fontsize=11)            
            plt.savefig(output_directory+filename+'_{:06d}.png'.format(count), dpi=dpi, bbox_inches='tight',
                                figsize=figsize)
            plt.close()
            count += 1


class MultiMovieBuddy():

    def __init__(self, dirs, max_files, start_files):
        buddies = []

        for i, dir in enumerate(dirs):
            buddies.append(MovieBuddy(dir, max_files=max_files[i], start_file=start_files[i]))

        self.buddies = buddies

    def add_subplot(self, *args, **kwargs):
        for buddy in self.buddies:
            buddy.add_subplot(*args, **kwargs)

    def define_plot_grid(self, rows=None, cols=1):
        if rows == None:
            rows = len(self.buddies)/cols
        self.rows = rows
        self.cols = cols

    def analyze_movie_subplots(self):
        for buddy in self.buddies:
            buddy.analyze_movie_subplots()

    def make_plots(self, outdir, filename='movie_montage', dpi=200, figsize=(10,10), bare=False):
        
        plot_slices = []
        for i, buddy in enumerate(self.buddies):
            if not hasattr(buddy, 'local_files'):
                continue
            profs = buddy.grab_whole_profile(buddy.local_files['slices'], buddy.local_writes_per_file, \
                        subkey=['tasks']*len(buddy.fields_to_grab), profile_name=buddy.fields_to_grab)
            slices = profs[buddy.fields_to_grab[0]]
            try:
                slices -= buddy.ax[0]['t_avg']
            except:
                slices *= 1
            plot_slices.append(slices)


        for j in range(plot_slices[0].shape[0]):
            try:
                fig = plt.figure(figsize=figsize)
                save=True
                for i, buddy in enumerate(self.buddies):
                    if not hasattr(buddy, 'local_files') or len(plot_slices) < i+1:
                        save=False
                        break

                    ax = plt.subplot2grid((self.rows, self.cols), (int(np.floor(i/self.cols)), int(np.mod(i, self.cols))), \
                                colspan=1, rowspan=1)
                    buddy.plot_field(buddy.xs, buddy.zs, plot_slices[i][j,:,:], ax, buddy.ax[0]['field'], 
                        cmap='RdBu_r', min=-buddy.ax[0]['stdev'], max=buddy.ax[0]['stdev'], bare=bare)
                if save:
                    logger.info('saving fig {}/{}'.format(j+1, plot_slices[0].shape[0]))
                    fig.savefig(output_directory+'/'+filename+'_{:06d}.png'.format(j+1+buddy.cw_rank*20), dpi=dpi, figsize=figsize, bbox_inches='tight')
                plt.close()
            except:
                plt.close()


