#http://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
# has great notes on how to manipulate matplotlib legends.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy import stats
import dedalus.public as de

import h5py
import os

from mpi4py import MPI
MPI_COMM_WORLD = MPI.COMM_WORLD

import logging
logger = logging.getLogger(__name__.split('.')[-1])

COLORS=('darkslategray', 'fuchsia', 'orangered', \
        'firebrick', 'indigo', 'lightgoldenrodyellow', 'darkmagenta',\
        'turquoise', 'mediumvioletred', 'mediumorchid', 'saddlebrown',\
        'slateblue', 'darkcyan')
STYLES=('-', '--', '-.', ':')
GREY=(0.8,0.8,0.8)

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_power(ux):
    '''Calculates the power spectrum of a velocity field'''
    fft = np.fft.fft(ux, axis=len(ux.shape)-2)
    power = np.abs(fft)**2
    return np.real(power)

def get_l2_ends(field_l2_values):
    field_l2_ends = np.zeros(field_l2_values.shape[0])
    for i in range(field_l2_values.shape[0]):
        if len(np.where(field_l2_values[i,:] != 0)[0]) == 0:
            field_l2_ends[i] = 1
            continue
        field_l2_ends[i] = \
            field_l2_values[i, np.where(field_l2_values[i,:] != 0)[0][-1]]
    return field_l2_ends


class PlotBuddy:

    def __init__(self, root_dir, max_files=1e6, start_file=1,\
                    file_dirs=['slices', 'scalar', 'profiles'], comm=MPI_COMM_WORLD,
                    file_tag = 'FC_poly'):
        ''' 
            Initializes a plotting tool which grabs data from root_dir.

            PARAMETERS:
                root_dir    - Base directory of run file output 
                                        (contains checkpoints, slices, etc.)
                max_files   - The maximum number of files to analyze
                start_file  - The number of the first file to analyze
                                        (1==> *_s1.h5)
                file_dirs   - A list of file directories to read in
                comm        - MPI communication group used for plotting
                file_tag    - The tag at the beginning of a file
        '''

        if root_dir[-1] != '/':
            root_dir += '/'
        self.root_dir       = root_dir
        self.start_file     = start_file
        self.file_dirs      = file_dirs
        self._read_atmosphere()
        self._derive_atmosphere_parameters()
        self._get_all_files()

        self.cw_size        = comm.Get_size()
        self.cw_rank        = comm.Get_rank()
        self.n_files_total  = len(self.global_files[file_dirs[0]])

        if self.n_files_total > max_files:
            self.n_files_total = max_files
            for dir in self.file_dirs:
                self.global_files[dir] = self.global_files[dir][:max_files]

        logger.info('Getting {} files from {:s}'.\
                        format(self.n_files_total,self.root_dir))

        #Get all of the files associated with this process and figure out how 
        # many files are below.
        self.n_files = int(np.floor(self.n_files_total/self.cw_size))
        if self.cw_rank < np.mod(self.n_files_total, self.cw_size) \
                and self.cw_size > 1:
            self.n_files += 1
            files_below = self.cw_rank * self.n_files
        elif self.cw_size == 1:
            files_below = 0
        else:
            files_below = (self.n_files+1)*\
                            np.mod(self.n_files_total, self.cw_size) +\
                           (self.cw_rank - np.mod(self.n_files_total,\
                            self.cw_size))*self.n_files
        if self.n_files == 0:
            self.idle = True
            self.comm = None
            self.cw_rank = -1
            self.cw_size = -1
            return
        else:
            self.idle = False
            n_group = self.n_files_total
            if n_group > self.cw_size:
                n_group = self.cw_size
            self.comm = comm.Create(comm.Get_group().Incl(np.arange(n_group)))
            self.cw_rank = self.comm.rank
            self.cw_size = self.comm.size

        self.files_below = int(files_below)
        self.file_number_start = self.files_below + 1
        self.file_number_end = self.files_below +  self.n_files
        
        upper = self.n_files + self.files_below
        lower = self.files_below

        self.local_files = dict()
        for dir in self.file_dirs:
            self.local_files[dir] = self.global_files[dir][lower:upper]

        #Figure out how many writes there are per file and get x, z.
        self._get_basic_info()
        self._get_global_info()

        self.ax = []


    def _derive_atmosphere_parameters(self):
        if self.atmosphere['atmosphere_name'] == 'single polytrope':
            chi_top = np.sqrt(self.atmosphere['prandtl']*(self.atmosphere['Lz']**3*\
                    np.abs(self.atmosphere['delta_s_atm']/self.atmosphere['Cp'])*\
                    self.atmosphere['g'])/self.atmosphere['rayleigh'])
            self.atmosphere['rayleigh_bottom']  = self.atmosphere['rayleigh']*\
                    np.exp(self.atmosphere['n_rho_cz'])
            self.atmosphere['t_buoy']           = np.sqrt(self.atmosphere['Lz']/\
                    (self.atmosphere['g']*self.atmosphere['epsilon']))
            self.atmosphere['t_therm']          = 2*self.atmosphere['Lz']**2 /\
                    (self.atmosphere['chi'][int(self.atmosphere['chi'].shape[0]/2)-1]+\
                            self.atmosphere['chi'][int(self.atmosphere['chi'].shape[0]/2)])


    def _read_atmosphere(self):
        '''
        Reads atmospheric parameters from the file root_dir/atmosphere/atmosphere.h5
        '''
        file_name = self.root_dir + '/atmosphere/atmosphere.h5'
        f = h5py.File('{:s}'.format(file_name))
        self.atmosphere = dict()
        for key in f.keys():
            if f[key].shape == ():
                self.atmosphere[key] = f[key].value
            else:
                self.atmosphere[key] = f[key][:]
        nu_top = np.sqrt(self.atmosphere['prandtl']*(self.atmosphere['Lz']**3 * \
                        np.abs(self.atmosphere['delta_s_atm']/self.atmosphere['Cp'])*\
                            self.atmosphere['g'])/self.atmosphere['rayleigh'])
        chi_top = nu_top/self.atmosphere['prandtl']
        self.atmosphere['chi'] = chi_top/self.atmosphere['rho0']
        self.atmosphere['nu'] = nu_top/self.atmosphere['rho0']

    def _get_all_files(self):
        '''
            Gets all of the .h5 files in root_dir.  Only gets slices,
            profiles, and scalar files.
        '''
        self.global_files = dict()
        for dir in self.file_dirs:
            self.global_files[dir] = self._get_files(dir)


    def _get_files(self, specific_dir):
        '''
            Gets all of the .h5 files from sub-directories in root_dir.
            Usually, specific_dir = ['slices', 'profiles', 'scalar']
        '''
        file_list = []
        for file in os.listdir(self.root_dir+specific_dir):
            if file.endswith('.h5'):
                if int(file.split('.')[0].split('_')[-1][1:]) < self.start_file:
                    continue
                file_list.append([self.root_dir+specific_dir+'/'+file, \
                     int(file.split('.')[0].split('_')[-1][1:])])
      
        file_list = sorted(file_list, key=lambda x: x[1])
        return file_list
       
    def _get_basic_info(self):
        '''
            Grabs some important quantities for the local processor.  Gets
            x, z, t.
        '''
        if self.idle: return
        self.x, self.z, self.xs, self.zs = None, None, None, None
        if 'slices' in self.file_dirs:
            f = h5py.File("{:s}".format(self.global_files['slices'][0][0]))
            self.x = np.array(f['scales']['x']['1.0'][:], dtype=np.float32)
            self.z = np.array(f['scales']['z']['1.0'][:], dtype=np.float32)
            self.zs, self.xs = np.meshgrid(self.z,self.x)
            f.close()
        elif 'profiles' in self.file_dirs:
            f = h5py.File("{:s}".format(self.global_files['profiles'][0][0]))
            self.z = np.array(f['scales']['z']['1.0'][:], dtype=np.float32)
            self.nz = self.z.shape[0]
            self.Lz = np.max(self.z)
            f.close()

    def _get_global_info(self):
        '''
            Calculates the number of writes in each .h5 file, as well as what
            time each file starts and ends.
        '''
        if self.idle: return
        file_writes_each = np.zeros(self.n_files_total, dtype=np.float32)
        file_start_times = np.zeros(self.n_files_total, dtype=np.float32)
        file_end_times   = np.zeros(self.n_files_total, dtype=np.float32)
        file_writes = []
        local_times = []
        
        index = self.files_below
        my_weirds = 0
        my_weird_indices = []
        for i, item in enumerate(self.local_files[self.file_dirs[0]]):
            f = h5py.File("{:s}".format(item[0]), flag='r')
            t = np.array(f['scales']['sim_time'][:], dtype=np.float32)
            f.close()
            if '_s1.h5' in item[0] and t[0] == 0:
                t[0] += 1e-6
            #Filter out weird writes e.g. between checkpoints
            good_t = np.where(t != 0)
            if len(good_t[0]) != len(t):
                values = []
                bad_indices = []
                for j in range(len(t)):
                    if np.isnan(t[j]):
                        bad_indices.append(j)
                        continue
                    elif t[j] not in values:
                        values.append(t[j])
                    else:
                        bad_indices.append(j)
                good_indices = []
                for j in range(len(good_t[0])):
                    if good_t[0][j] not in bad_indices:
                        good_indices.append(good_t[0][j])
                t = t[good_indices]
                for dir in self.file_dirs:
                    self.local_files[dir][i].append(good_indices)
                    self.global_files[dir][index].append(good_indices)
                my_weirds += 1
                my_weird_indices.append(index)

            file_writes.append(t.shape[0])
            file_writes_each[index] = t.shape[0]
            file_start_times[index] = t[0]
            file_end_times[index]   = t[-1]
            for i in range(t.shape[0]):
                local_times.append(t[i])
            index += 1

        local_times = np.array(local_times, dtype=np.float32)

        #Get info about all file writes, start times, end times
        global_file_writes = np.zeros(self.n_files_total, dtype=np.float32)
        global_start_times = np.zeros(self.n_files_total, dtype=np.float32)
        global_end_times = np.zeros(self.n_files_total, dtype=np.float32)
        self.comm.Allreduce(file_writes_each, global_file_writes, op=MPI.MAX)
        self.comm.Allreduce(file_start_times, global_start_times, op=MPI.MAX)
        self.comm.Allreduce(file_end_times, global_end_times, op=MPI.MAX)

        #Get all info about total file times
        length = np.zeros(1)
        full_length = np.zeros(1)
        length[0] = len(local_times)
        self.comm.Allreduce(length, full_length, op=MPI.SUM)
        full_times = np.zeros(full_length[0], dtype=np.float32)
        my_times = np.zeros(full_length[0], dtype=np.float32)
        my_times[np.sum(global_file_writes[:self.files_below]):\
                    np.sum(global_file_writes[:self.files_below+self.n_files])]\
                        = local_times
        self.comm.Allreduce(my_times, full_times, op=MPI.SUM)

        #Store time information away in the class
        self.local_times        = local_times
        self.global_times       = full_times
        self.local_start_times  = file_start_times - global_start_times[0]
        self.global_start_times = global_start_times  - global_start_times[0]
        self.local_end_times    = file_end_times  - global_start_times[0]
        self.global_end_times   = global_end_times - global_start_times[0]
        self.sim_start_time     = global_start_times[0]


        
        self.global_writes_per_file  = global_file_writes
        self.local_writes_per_file   = np.array(file_writes, dtype=np.float32)
        self.max_writes_per_file     = np.max(global_file_writes)
        self.global_total_writes     = int(np.sum(global_file_writes))
        self.local_total_writes      = int(np.sum(self.local_writes_per_file))
        self.local_writes_below      = int(np.sum(self.global_writes_per_file\
                                                        [:self.files_below]))

        #Communicate weird files
        weirds_local = np.zeros(self.cw_size, dtype=np.int16)
        weirds_global  = np.zeros(self.cw_size, dtype=np.int16)
        weirds_local[self.cw_rank] = my_weirds
        self.comm.Allreduce(weirds_local, weirds_global, op=MPI.SUM)
        important_indices = np.where(weirds_global != 0)[0]
        for i in range(len(important_indices)):
            for j in range(weirds_global[important_indices[i]]):
                local_index=np.zeros(1, dtype=np.int16)
                global_index=np.zeros(1, dtype=np.int16)
                local_good_indices = np.zeros(self.max_writes_per_file, dtype=np.int16)
                global_good_indices = np.zeros(self.max_writes_per_file, dtype=np.int16)
                if important_indices[i] == self.cw_rank:
                    print('process {:d} is communicating a weird file'.format(self.cw_rank))
                    local_good_indices -= 1
                    local_index[0] = my_weird_indices[j]
                    good_writes = np.array(self.global_slices_files[local_index[0]][2])
                    local_good_indices[:good_writes.shape[0]] = good_writes
                self.comm.Allreduce(local_good_indices, global_good_indices, op=MPI.SUM)
                self.comm.Allreduce(local_index, global_index, op=MPI.SUM)
                good_indices = global_good_indices[np.where(global_good_indices != -1)]
                if important_indices[i] != self.cw_rank:
                    for dir in self.file_dirs:
                        self.global_files[dir][global_index[0]].append(good_indices)


    def grab_whole_profile(self, file_list, writes_per_file, subkey=['tasks'], profile_name=['s']):
        '''
            Grabs the save field(s) specified in profile_name which have the
            subkey(s) specified in subkey.  Specify the time of file using
            slices, profiles, or scalar = True.
        '''
        if self.idle: return
        bigger_fields = dict()
        count = 0
        for i, file in enumerate(file_list):
            logger.info('opening file {}'.format(file[0]))
            f = h5py.File("{:s}".format(file[0]), flag='r')
            for j in range(len(profile_name)):
                field = np.array(f[str(subkey[j])][str(profile_name[j])], dtype=np.float32)
                if i == 0:
                    #Create space in memory for fields from all files
                    shape = list(field.shape)
                    shape[0] = np.sum(writes_per_file)
                    bigger_fields[profile_name[j]] = np.zeros(shape, dtype=np.float32)
                if len(file) > 2:
                    bigger_fields[profile_name[j]][count:count+writes_per_file[i]] =\
                            field[file[2]]
                else:
                    bigger_fields[profile_name[j]][count:count+writes_per_file[i]] =\
                            field
            count += writes_per_file[i]
            f.close()
        return bigger_fields


       

 
    def setup_plot_grid(self, nrows=1, ncols=1):
        '''
        Records the total number of rows and columns each figure 
        produced by the class should be.  Parameters specify number of
        rows and also number of columns.
        '''
        if self.idle: return
        self.nrows      = nrows 
        self.ncols      = ncols
        self.fig_dims   = (self.nrows, self.ncols)
