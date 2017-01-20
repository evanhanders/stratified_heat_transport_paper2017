from plot_buddy_base import *

class ScalarBuddy(PlotBuddy):

    def __init__(self, *args, **kwargs):
        super(ScalarBuddy, self).__init__(*args, **kwargs)
        self.tracked_scalars = dict()
        self.track_scalar('sim_time', subkey='scales')
        self.ax = []

        if self.cw_rank != 0:
            self.idle = True
        else:
            self.local_files = self.global_files
            self.local_writes_per_file = self.global_writes_per_file

    def track_scalar(self, scalar, subkey='tasks'):
        self.tracked_scalars[scalar] = subkey

    def pull_tracked_scalar(self):
        key_list = []
        subkey_list = []
        for key in self.tracked_scalars.keys():
            key_list.append(key)
            subkey_list.append(self.tracked_scalars[key])

        self.pulled_scalars = self.grab_whole_profile(  self.local_files['scalar'],\
                                                        self.local_writes_per_file,\
                                                        subkey=subkey_list,\
                                                        profile_name=key_list)
        for key in self.pulled_scalars.keys():
            if len(self.pulled_scalars[key].shape) > 1:
                self.pulled_scalars[key] = self.pulled_scalars[key].flatten()

    def add_subplot(self, rindex, cindex, x_value, y_value, log_x=False, log_y=False):
        if self.idle: return
        
        self.ax.append(dict())
        self.ax[-1]['position'] = (rindex, cindex)
        self.ax[-1]['x']        = x_value
        self.ax[-1]['y']        = y_value
        self.ax[-1]['log_x']    = log_x
        self.ax[-1]['log_y']    = log_y

    def make_plots(self, figsize=None, outdir='scalar_plots', filename='scalar_plots', dpi=300):
        '''
            Create all of the plots!
        '''
        if self.idle: return

        total_width = 0
        total_height = 0
        for i, ax in enumerate(self.ax):
            if ax['position'][0] + 1 > total_height:
                total_height = ax['position'][0] + 1
            if ax['position'][1] + 1 > total_width:
                total_width = ax['position'][1] + 1
        self.plot_dimensions = (total_height, total_width) #rows, ncols
        if figsize == None:
            figsize = (self.plot_dimensions[1]*(2*self.atmosphere['aspect_ratio']), self.plot_dimensions[0]*2+0.5)

        output_directory = self.root_dir + '/' + outdir + '/'
        if self.cw_rank == 0 and not os.path.exists('{:s}'.format(output_directory)):
            os.mkdir('{:s}'.format(output_directory))
        logger.info('saving figures to {}'.format(output_directory))

        fig = plt.figure(figsize=figsize)
        axes = dict()
        for i, ax in enumerate(self.ax):
            logger.info('Plotting {} v {}'.format(ax['y'], ax['x']))
            if ax['position'] in axes.keys():
                axis = axes[ax['position']]
            else:
                axis = plt.subplot2grid(self.plot_dimensions, ax['position'])
                axes[ax['position']] = axis

            axis.plot(self.pulled_scalars[ax['x']], self.pulled_scalars[ax['y']])
            axis.set_xlabel(ax['x'])
            axis.set_ylabel(ax['y'])
            if ax['log_x']:
                axis.set_xscale('log')
            if ax['log_y']:
                axis.set_yscale('log')

        plt.savefig(output_directory+filename, dpi=dpi, bbox_inches='tight', figsize=figsize)
        plt.close()


        
