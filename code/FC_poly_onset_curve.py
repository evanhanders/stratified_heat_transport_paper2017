"""
Dedalus script for finding the onset of compressible convection in a 
polytropic atmosphere

Usage:
    FC_poly_onset_curve.py [options] 

Options:
    --rayleigh_start=<Rayleigh>         Rayleigh number start [default: 1e-2]
    --rayleigh_stop=<Rayleigh>          Rayleigh number stop [default: 1e3]
    --rayleigh_steps=<steps>            Integer number of steps between start 
                                         and stop Ra   [default: 20]
    --kx_start=<kx_start>               kx to start at [default: 0.01]
    --kx_stop=<kx_stop>                 kx to stop at  [default: 10]
    --kx_steps=<kx_steps>               Num steps in kx space [default: 20]

    --3D
    --ky_start=<ky_start>               ky to start at [default: 0.01]
    --ky_stop=<ky_stop>                 ky to stop at  [default: 10]
    --ky_steps=<ky_steps>               Num steps in ky space [default: 20]

    --nz=<nz>                           z (chebyshev) resolution [default: 48]

    --n_rho_cz_start=<n_rho_cz_start>   Density scale heights (smallest run)
    --n_rho_cz_stop=<n_rho_cz_stop>     Density scale heights (largest run) 
    --n_rho_cz_steps=<n_rho_cz_steps>   Num steps in n_rho space
    --epsilon_start=<n_rho_cz_start>    Epsilon (smallest run) 
    --epsilon_stop=<n_rho_cz_stop>      Epsilon (largest run)  
    --epsilon_steps=<n_rho_cz_steps>    Number of epsilon steps to take 
    --epsilons=<epsilon>                A comma-delimited list of epsilons

    --bcs=<bcs>                         Boundary conditions ('fixed', 'mixed', 
                                            or 'flux') [default: fixed]
    --n_rho_cz_default=<n_rho>          Default nrho [default: 3]
    --epsilon_default=<epsilon>         Default eps  [default: 0.5]
    --gamma_default=<gamma>             Default gamma [default: 5/3]
    --constant_chi                      If true, use const chi
    --constant_nu                       If true, use const nu

    --rotating                          If true, add rotation
    --taylor_default=<Ta>               The Taylor number [default: 1e4]
    --angle_default=<angle>             The angle of Omega [default: 0]

    --out_dir=<out_dir>                 Base output dir [default: ./]
"""
from docopt import docopt
from onset_solver import OnsetSolver

args = docopt(__doc__)

file_name = 'FC_poly_onsets'

##########################################
#Set up ra / kx spans
ra_start = float(args['--rayleigh_start'])
ra_stop = float(args['--rayleigh_stop'])
ra_steps = float(args['--rayleigh_steps'])

kx_start = float(args['--kx_start'])
kx_stop = float(args['--kx_stop'])
kx_steps = float(args['--kx_steps'])

ky_start = float(args['--ky_start'])
ky_stop = float(args['--ky_stop'])
ky_steps = float(args['--ky_steps'])

threeD = args['--3D']

file_name += '_ra{:s}-{:s}-{:s}_kx{:s}-{:s}-{:s}'.format(
                args['--rayleigh_start'],
                args['--rayleigh_stop'],
                args['--rayleigh_steps'],
                args['--kx_start'],
                args['--kx_stop'],
                args['--kx_steps'])

if threeD:
    file_name += '_ky{:s}-{:s}-{:s}'.format(
                args['--ky_start'],
                args['--ky_stop'],
                args['--ky_steps'])
    ky_list = (ky_start, ky_stop, ky_steps, True)
else:
    ky_list = None

##########################################
#Set up defaults for the atmosphere
const_kap = True
if args['--constant_chi']:
    const_kap = False
const_mu = True
if args['--constant_nu']:
    const_mu = False

nz = int(args['--nz'])
n_rho_default = float(args['--n_rho_cz_default'])
epsilon_default = float(args['--epsilon_default'])
try:
    gamma_default = float(args['--gamma_default'])
except:
    from fractions import Fraction
    gamma_default = float(Fraction(args['--gamma_default']))


defaults = {'n_rho_cz': n_rho_default,
            'epsilon':  epsilon_default,
            'constant_kappa': const_kap,
            'constant_mu':    const_mu,
            'nz':             nz,
            'gamma':          gamma_default}

##############################################
#Setup default arguments for equation building
if args['--rotating']:
    angle = float(args['--angle_default'])
    taylor= float(args['--taylor_default'])
    file_name += '_Ta{}_theta{}'.format(taylor, angle)
else:
    angle, taylor = 0, None
eqn_defaults_args = [1] #prandtl number
eqn_defaults_kwargs = {'Taylor': taylor,
                       'theta':  angle}

############################################
#Set up BCs
bcs = args['--bcs']
fixed_flux = False
fixed_T    = False
mixed_T    = False
if bcs == 'mixed':
    mixed_T = True
elif bcs == 'fixed':
    fixed_T = True
else:
    fixed_flux = True

bc_defaults = {'fixed_temperature': fixed_T,
               'mixed_flux_temperature': mixed_T,
               'fixed_flux': fixed_flux,
               'stress_free': True}
file_name += '_bc_{:s}'.format(bcs)

#####################################################
#Initialize onset solver
solver = OnsetSolver(
            eqn_set=0, atmosphere=0, 
            ra_steps=(ra_start, ra_stop, ra_steps, True),
            kx_steps=(kx_start, kx_stop, kx_steps, True),
            ky_steps=ky_list,
            atmo_kwargs=defaults,
            eqn_args=eqn_defaults_args,
            eqn_kwargs=eqn_defaults_kwargs,
            bc_kwargs=bc_defaults)

###############################################
#Add tasks (nrho, epsilon) in case we need to solve over
#those parameter spaces, too
if args['--n_rho_cz_start'] != None\
        and args['--n_rho_cz_stop'] != None \
        and args['--n_rho_cz_steps'] != None:
    start = float(args['--n_rho_cz_start'])
    stop  = float(args['--n_rho_cz_stop'])
    steps = int(args['--n_rho_cz_steps'])
    log = False
    if stop/start > 10:
        log = True
    solver.add_task('n_rho_cz', start, stop, n_steps=steps, log=log)
    file_name += '_nrhocz{:s}-{:s}-{:s}'.format(
                    args['--n_rho_cz_start'],
                    args['--n_rho_cz_stop'],
                    args['--n_rho_cz_steps'])

if args['--epsilon_start'] != None\
        and args['--epsilon_stop'] != None \
        and args['--epsilon_steps'] != None:
    start = float(args['--epsilon_start'])
    stop  = float(args['--epsilon_stop'])
    steps = int(args['--epsilon_steps'])
    log = False
    if stop/start > 10:
        log = True
    solver.add_task('epsilon', start, stop, n_steps=steps, log=log)
    file_name += '_eps{:s}-{:s}-{:s}'.format(
                    args['--epsilon_start'],
                    args['--epsilon_stop'],
                    args['--epsilon_steps'])

#############################################
#Crit find!
out_dir = args['--out_dir']
tol = 1e-3
if threeD:
    tol = 5e-1
solver.find_crits(out_dir=out_dir, out_file='{:s}.h5'.format(file_name), tol=tol)

############################################
#plot and save
solver.plot_onset_curves(out_dir=out_dir, fig_name='{:s}.png'.format(file_name))
