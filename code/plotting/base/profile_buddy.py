from plot_buddy_base import *
import matplotlib.ticker as ticker
from scipy.stats import mode


class ProfileBuddy(PlotBuddy):
    
    def __init__(self, dir, write_cadence=None, outdir='profile_plots', **kwargs):
        super(ProfileBuddy, self).__init__(dir, **kwargs)
        if self.idle: return
        self.write_cadence = write_cadence #in t_buoy
        self.outdir = outdir
        self.output_dir = self.root_dir + '/' + outdir + '/'

    def add_subplot(self, rindex, cindex, hlabel, vlabel, field,\
                        colspan=1, rowspan=1):
        '''
            rindex -- row index of the subplot, starting from zero
            cindex -- column index of the subplot, starting from zero
            hlabel -- the label for the horizontal axis 
                                            ('x', 't', or something else)
            vlabel -- the label for the vertical axis ('z', or something else)
            field  -- the field to grab and plot (name must match field in file)
            colspan  -- The number of columns that the subplot spans across
            rowspan  -- The number of rows that the subplot spans across
        '''
        if self.idle: return
        #Here's where we specify the different types of plots -- so far only 
        # expands a few of the colormaps.
        
        subplot = dict()
        subplot['field']            = field
        subplot['plot_index']       = list((rindex, cindex)) #row, column
        subplot['colspan']            = colspan #colspan
        subplot['rowspan']            = rowspan #rowspan
        subplot['xlabel']           = hlabel
        subplot['ylabel']           = vlabel

        self.ax.append(subplot)


    def create_comm_world(self):
        '''
        Creates a smaller comm group that is only the size of the number of l2_avgs that
        are going to be done in this run
        '''
        if self.idle: return
        file_groups = np.floor(self.global_end_times/(self.write_cadence*self.atmosphere['t_buoy']))
        n_procs_max = int(np.max(file_groups)+1)
        my_files_ind= (file_groups % self.cw_size) == self.cw_rank

        self.global_average_times = []
        self.local_average_start_times = []
        self.local_average_end_times = []
        #use start time within each file group as average time
        self.local_writes_per_plot = []
        self.l2_writes_per_file = []
        for i in range(len(file_groups)):
            if i == 0 or file_groups[i] != file_groups[i-1]:
                self.global_average_times.append(self.global_start_times[i])
            if (i == 0 and my_files_ind[i]) or (my_files_ind[i] and not my_files_ind[i-1]):
                self.local_writes_per_plot.append(self.global_writes_per_file[i])
                self.local_average_start_times.append(self.global_start_times[i])
                if (i == len(file_groups)-1 and  my_files_ind[i]) or (my_files_ind[i] and not my_files_ind[i+1]):
                    self.local_average_end_times.append(self.global_end_times[i])
            elif (i == len(file_groups)-1 and  my_files_ind[i]) or (my_files_ind[i] and not my_files_ind[i+1]):
                self.local_writes_per_plot[-1] += self.global_writes_per_file[i]
                self.local_average_end_times.append(self.global_end_times[i])
            elif my_files_ind[i]:
                self.local_writes_per_plot[-1] += self.global_writes_per_file[i]
            if my_files_ind[i]:
                self.l2_writes_per_file.append(self.global_writes_per_file[i])
                
        self.my_l2_files    = dict()
        for d, dir in enumerate(self.file_dirs):
            files = []
            for i,ind in enumerate(np.where(my_files_ind == True)[0]):
                files.append(self.global_files[dir][ind])
            self.my_l2_files[dir] = files
        self.l2_writes_per_file = np.array(self.l2_writes_per_file)

        if n_procs_max >= self.cw_size:
            self.l2_comm  = self.comm
            self.l2_rank  = self.cw_rank
            self.l2_size  = self.cw_size
        else:
            if self.cw_rank >= n_procs_max:
                self.doing_l2 = False
            else:
                self.doing_l2 = True
            self.l2_comm = self.comm.Create(self.comm.Get_group().Incl(np.arange(n_procs_max)))
            self.l2_rank = self.l2_comm.rank
            self.l2_size = self.l2_comm.size

        logger.info('breaking up L2 norms across {} processes'.format(self.l2_size))

    def calculate_l2_norm(self, field, full_l2_fields, sum=False):
        if self.idle: return
        max_l2_writes = 0
        average_profiles = []
        l2_norms = []
        for i, writes in enumerate(self.local_writes_per_plot):
            writes_below = np.sum(self.local_writes_per_plot[:i])
            writes_after = np.sum(self.local_writes_per_plot[:i+1])
            current_writes = full_l2_fields[field][writes_below:writes_after]
            avg = np.zeros((1,current_writes.shape[-1]), dtype=np.float32)
            l2_norm = np.zeros(current_writes.shape[0], dtype=np.float32)
            if sum:
                for j in range(current_writes.shape[0]):
                    sum_array = np.zeros((1,current_writes.shape[-1]), dtype=np.float32)
                    for plot in self.ax:
                        plot_type = plot['type']
                        field = plot['field']
                        filetype = plot['filetype']
                        sum_plot = plot['sum']
                        if plot_type != 'l2_avg':
                            continue
                        if sum_plot:
                            continue
                                       
                        current_writes = full_l2_fields\
                                    [field][writes_below:writes_after]
                        sum_array += current_writes[j,:]

                    old_avg = np.copy(avg)
                    avg *= j
                    avg += sum_array
                    avg /= j+1
                    l2_norm[j] = np.linalg.norm(old_avg-avg, ord=2)/ \
                                    np.linalg.norm(old_avg, ord=2)
            else:
                for j in range(current_writes.shape[0]):
                    old_avg = np.copy(avg)
                    avg *= j
                    avg += current_writes[j,:]
                    avg /= j+1

                    if j == 0:
                        continue
                    l2_norm[j] = np.linalg.norm(old_avg-avg, ord=2)/ \
                                np.linalg.norm(old_avg, ord=2)
                    if j == 1:
                        l2_norm[0] = l2_norm[j]
            l2_package = [np.arange(current_writes.shape[0]), l2_norm]
            
            l2_norms.append(l2_package)
            average_profiles.append(avg)
            
            if current_writes.shape[0] > max_l2_writes:
                max_l2_writes = current_writes.shape[0]

        if max_l2_writes > self.max_l2_writes:
            self.max_l2_writes = max_l2_writes

        return l2_norms, average_profiles

    def analyze_subplots(self):
        if self.idle: return
        self.l2_norms = dict()
        self.average_profiles = dict()

        #Break up files into groups and learn about how much time each average covers, etc.
        self.create_comm_world()

        #Get the names of all profiles we need
        profile_names,  profile_subkeys = [], []
        derived_keys = []
        for plot in self.ax:
            #Get plot info
            field = plot['field']
            profile_names.append(field)
            profile_subkeys.append('tasks')

            #Get all profiles
        full_l2_fields = self.grab_whole_profile(   self.my_l2_files['profiles'], \
                                                    self.l2_writes_per_file, \
                                                    subkey=profile_subkeys, \
                                                    profile_name=profile_names )
        self.max_l2_writes = 0
        for plot in self.ax:
            field = plot['field']
            logger.info('analyzing {:s}'.format(field))
            l2_norms, average_profiles = self.calculate_l2_norm(field, full_l2_fields, sum=False)
            self.l2_norms[field] = l2_norms
            self.average_profiles[field] = np.array(average_profiles, dtype=np.float32)

        num_rows, num_cols = 0, 0
        for plot in self.ax:
            plot_index = plot['plot_index']
            if plot_index[0]+1 > num_rows:
                num_rows = plot_index[0]+1
            if plot_index[1]+1 > num_cols:
                num_cols = plot_index[1]+1
        self.fig_dims = (num_rows + 1, num_cols)
        logger.info('done with analyzing')

    def communicate_profiles(self):
        if self.idle: return
        comm_dicts = [self.average_profiles, self.l2_norms]
        self.full_average_profiles, self.full_l2_norms = dict(), dict()
                    
        for k, dictionary in enumerate(comm_dicts):
            for i, key in enumerate(sorted(list(dictionary.keys()))):
                logger.info('communicating {:s}'.format(key))
                if k == 1:
                    local_array = np.array(dictionary[key])[:,1]
                    save_key = key+'_l2'
                else:
                    local_array =  np.array(dictionary[key])
                    save_key = key
                if i == 0:
                    #Figure out how big of an array we need
                    n_writes_local = np.zeros(1)
                    n_writes_global = np.zeros(1)
                    n_writes_local[0] = local_array.shape[0]

                    self.l2_comm.Allreduce(n_writes_local, n_writes_global, op=MPI.SUM)
                    if k == 1:
                        l2_writes_global = np.zeros(1)
                        n_writes_local[0] = self.max_l2_writes
                        self.l2_comm.Allreduce(n_writes_local, l2_writes_global,\
                                            op = MPI.MAX)

                shape = list(local_array.shape)
                shape[0] = n_writes_global[0]
                if k == 1:
                    shape = [n_writes_global[0], l2_writes_global[0]]
                this_profile = np.zeros(shape, dtype=np.float32)
                for j in range(local_array.shape[0]):
                    index = j*self.l2_size+self.l2_rank
                    this_profile[index,:local_array[j].shape[0]] = local_array[j]
                global_profile = np.zeros(shape, dtype=np.float32)
                self.l2_comm.Allreduce(this_profile, global_profile, op=MPI.SUM)
                if k == 0:
                    self.full_average_profiles[save_key] = global_profile
                elif k == 1:
                    self.full_l2_norms[save_key] = global_profile
      
    def save_profiles(self,  filename='profiles'):
        if self.idle: return
        if self.l2_rank == 0:
            if self.output_dir[-1] != '/':
                self.output_dir += '/'
            if self.l2_rank == 0 and not os.path.exists('{:s}'.format(self.output_dir)):
                os.mkdir('{:s}'.format(self.output_dir))
            f = h5py.File('{:s}/{:s}.h5'.format(self.output_dir, filename), 'w')
            f2 = open('{:s}/{:s}_keys.txt'.format(self.output_dir, filename), 'w')
            f2.write('STARTING AVERAGE KEYS \n')
            f2.write('--------------------- \n')

            z_basis = de.Chebyshev('z', self.atmosphere['nz'], interval=[0., self.atmosphere['Lz']], dealias=3/2)
            domain  = de.Domain([z_basis], grid_dtype=np.float64, comm=MPI.COMM_SELF)
            
            current_field = domain.new_field()
            work_field    = domain.new_field()

            for key in self.full_average_profiles.keys():
                means, stdevs = [], []
                for i in range(self.full_average_profiles[key].shape[0]):
                    current_field.set_scales(1, keep_data=False)
                    current_field['g'] = self.full_average_profiles[key][i,:]
                    current_field.antidifferentiate('z', ('left', 0), out=work_field)
                    mean = work_field.interpolate(z=self.atmosphere['Lz'])['g'][0]/self.atmosphere['Lz']
                    current_field.set_scales(1, keep_data=False)
                    current_field['g'] = (self.full_average_profiles[key][i,:] - mean)**2
                    current_field.antidifferentiate('z', ('left', 0), out=work_field)
                    stdev = np.sqrt(work_field['g'][-1]/self.atmosphere['Lz'])

                    means.append(mean)
                    stdevs.append(stdev)

                f[key] = np.array(self.full_average_profiles[key], dtype=np.float32)
                f[key+'_mean'] = np.array(means)
                f[key+'_stdev']= np.array(stdevs)
                f2.write(key + '\n')
                f2.write(key + '_mean \n')
                f2.write(key + '_stdev \n')


            f2.write('\nSTARTING L2 NORM KEYS\n')
            f2.write(  '---------------------\n')
            for key in self.full_l2_norms.keys():
                f[key] = np.array(self.full_l2_norms[key], dtype=np.float32)
                f2.write(key + '\n')
            f2.write('\nSIMPLER ARRAYS:\n')
            f2.write('full_average_times\n')
            f2.write('z')
            f['full_average_times'] = np.array(self.global_average_times, dtype=np.float32)
            f['z'] = self.z
            f.close()
            f2.close()

        self.comm.Barrier()
            
            
    def make_plots(self, figsize=(16,8), \
                    write_number_start=1, clear_plots=True, dpi=300):
        '''
            Create all of the plots!
        '''
        if self.idle: return

        self.comm.Barrier()
        filename=self.outdir

        if self.cw_rank == 0 and not os.path.exists('{:s}'.format(self.output_dir)):
            os.mkdir('{:s}'.format(output_dir))
        logger.info('saving figures to {}'.format(self.output_dir))

        #We need to communicate all of the l2_avg profiles we're plotting
        if clear_plots == False and self.full_average_profiles == dict():
            self.communicate_profiles()

        min_max = dict()
        for i in range(len(self.local_average_start_times)):
            if hasattr(self, 'local_average_end_times'):
                plt.suptitle('t = {:1.3e}-{:1.3e}'.format(self.local_average_start_times[i], self.local_average_end_times[i]),x=0.84,y=1, fontsize=16)
            print('plotting figure {}/{} on process {}'.\
                        format(i, len(self.local_average_start_times), self.l2_rank))
            fig = plt.figure(figsize=figsize)
            known_plots = []
            for j, plot in enumerate(self.ax):
                img_axis = -1
                axis_index = -1
                field = plot['field']
                plot_ind = list(plot['plot_index'])

                #If we already have this axis created, find it
                for k in range(len(known_plots)):
                    if list((known_plots[k][0], known_plots[k][1])) ==\
                            plot_ind:
                        img_axis = known_plots[k][2]
                        known_plots[k][3] = True
                        axis_index = k
                if img_axis == -1:
                    known_plots.append(list((plot_ind)))
                colspan = plot['colspan']
                rowspan = plot['rowspan']
                hlabel = plot['xlabel']
                vlabel = plot['ylabel']

                #need to plot average and l2 norm
                if img_axis == -1:
                    avg_axis = plt.subplot2grid(self.fig_dims, plot_ind,\
                            colspan=colspan, rowspan=rowspan)
                    avg_axis.set_axis_bgcolor(GREY)
                    plot_ind[0] += rowspan
                    l2_axis = plt.subplot2grid(self.fig_dims, plot_ind, \
                            colspan=colspan, rowspan=rowspan)
                    l2_axis.set_axis_bgcolor(GREY)
                    axes = [avg_axis, l2_axis]
                    known_plots[-1].append(axes)
                    known_plots[-1].append(False)
                    known_plots[-1].append(0)
                    axis_index = len(known_plots)-1
                    legend=False
                else:
                    avg_axis = img_axis[0]
                    l2_axis = img_axis[1]
                    legend=True

                #Plot all the previous curves if need be.  Else just plot this one.
                if field == 'kappa_flux_fluc_z':
                    add = self.kappa*self.epsilon/self.Cp
                else:
                    add = 0
                if not clear_plots:
                    index = i * self.l2_size + self.l2_rank
                    for k in range(index):
                        avg_axis.plot(self.z, \
                                self.full_average_profiles\
                                [field][k][0,:]+add, label=field)
                avg_axis.plot(self.z, \
                        self.average_profiles[field][i][0,:]+add, \
                        label=field, color=COLORS[known_plots[axis_index][-1]], linewidth=2)
                l2_axis.plot(self.l2_norms[field][i][0], \
                        self.l2_norms[field][i][1], label=field,
                                color=COLORS[known_plots[axis_index][-1]], linewidth=2)
                known_plots[axis_index][-1] += 1

                try:
                    l2_axis.set_yscale('log')
                except:
                    l2_axis.set_yscale('linear')
                try:
                    l2_axis.set_xscale('log')
                except:
                    l2_axis.set_xscale('linear')

                if not legend:
                    avg_axis.set_xlim(np.min(self.z), np.max(self.z))
                    l2_axis.set_xlim(np.min(self.l2_norms[field][i][0]),\
                                    np.max(self.l2_norms[field][i][0]))

                    avg_axis.axhline(0, linestyle='--', color='black')

                    l2_axis.set_xlabel('number of writes')
                    l2_axis.set_ylabel('l2 norm')

                    avg_axis.set_xlabel(hlabel)
                    plt.xticks(rotation='vertical')
                    avg_axis.set_ylabel(vlabel)
                else:
                    avg_axis.legend(frameon=False, loc='upper left', fontsize=18)
                    l2_axis.legend(frameon=False, loc='lower left', fontsize=18)
                
            count = self.cw_rank + self.cw_size*i + write_number_start
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
            plt.savefig(self.output_dir+filename+'_{:04d}.png'.format(count))
            plt.close()

