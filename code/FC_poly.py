"""
Dedalus script for 2D or 3D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options] 

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --aspect=<aspect>                    Aspect ratio [default: 4]
     
    --restart=<restart_file>             Restart from checkpoint

    --nz=<nz>                            vertical z (chebyshev) resolution 
    --nz_cz=<nz>                         vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz_cz
    --ny=<ny>                            Horizontal y (Fourier) resolution; if not set, ny=nx (3D only)  
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3]
    --run_time=<run_time>                Run time, in hours [default: 23.5]

    --rk222                              Use RK222 as timestepper

    --3D                                 Do 3D run
    --mesh=<mesh>                        Processor mesh if distributing 3D run in 2D
        
    --MHD                                Do MHD run
    --MagneticPrandtl=<MagneticPrandtl>  Magnetic Prandtl Number = nu/eta [default: 1]

    --fixed_T                            Fixed Temperature boundary conditions (top and bottom)
    --fixed_Tz                           Fixed Temperature gradient boundary conditions (top and bottom)
        
    --label=<label>                      Additional label for run output directory

"""
import logging
logger = logging.getLogger(__name__)

import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing = True
except:
    do_checkpointing = False

def FC_polytrope(Rayleigh=1e6, Prandtl=1, MagneticPrandtl=1, aspect_ratio=4, MHD=False, n_rho_cz=3.5,
                      fixed_T=False, fixed_Tz=False,
                      rk222=False, threeD=False, mesh=None,
                      restart=None, nz=128, nx=None, ny=None,
                      data_dir='./', run_time=23.5):
    import numpy as np
    import time
    import equations
    import os
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx is None:
        nx = int(np.round(nz*aspect_ratio))
    if threeD and ny is None:
        ny = nx

    if MHD:
        atmosphere = equations.FC_MHD_polytrope(nx=nx, nz=nz, constant_kappa=True, n_rho_cz=n_rho_cz)
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, MagneticPrandtl)
    else:
        if threeD:
            atmosphere = equations.FC_polytrope_3d(nx=nx, ny=ny, nz=nz, aspect_ratio=aspect_ratio, mesh=mesh,
                                                   constant_kappa=True, constant_mu=True, n_rho_cz=n_rho_cz)
        else:
            atmosphere = equations.FC_polytrope_2d(nx=nx, nz=nz, aspect_ratio=aspect_ratio, constant_kappa=True, constant_mu=True, n_rho_cz=n_rho_cz)
        atmosphere.set_IVP_problem(Rayleigh, Prandtl)
    if fixed_T:
        atmosphere.set_BC(fixed_temperature=fixed_T)
    elif fixed_Tz:
        atmosphere.set_BC(fixed_flux=fixed_Tz)
    else:
        atmosphere.set_BC()

    problem = atmosphere.get_problem()

    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    if rk222:
        logger.info("timestepping using RK222")
        ts = de.timesteppers.RK222
        cfl_safety_factor = 0.2*2
    else:
        logger.info("timestepping using RK443")
        ts = de.timesteppers.RK443
        cfl_safety_factor = 0.2*4

    # Build solver
    solver = problem.build_solver(ts)

    if do_checkpointing:
        checkpoint = Checkpoint(data_dir)
        checkpoint.set_checkpoint(solver, wall_dt=1800)

    if restart is None:
        atmosphere.set_IC(solver)        
    else:
        logger.info("restarting from {}".format(restart))
        checkpoint.restart(restart, solver)

    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time,
                                                                      atmosphere.top_thermal_time))

    max_dt = atmosphere.buoyancy_time*0.25

    report_cadence = 1
    output_time_cadence = 0.1*atmosphere.buoyancy_time
    solver.stop_sim_time = 100*atmosphere.thermal_time
    solver.stop_iteration= np.inf
    solver.stop_wall_time = run_time*3600

    Hermitian_cadence = 100
    
    logger.info("output cadence = {:g}".format(output_time_cadence))

    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence)

    
    cfl_cadence = 1
    CFL = flow_tools.CFL(solver, initial_dt=max_dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.1)

    if threeD:
        CFL.add_velocities(('u', 'v', 'w'))
    else:
        CFL.add_velocities(('u', 'w'))
        
    if MHD:
        CFL.add_velocities(('Bx/sqrt(4*pi*rho_full)', 'Bz/sqrt(4*pi*rho_full)'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')
    if MHD:
        #flow.add_property("sqrt(Bx*Bx + Bz*Bz) / Rm", name='Lu')
        flow.add_property("abs(dx(Bx) + dz(Bz))", name='divB')
        flow.add_property("abs(dx(Ax) + dz(Az))", name='divA')

        Tobias_gambit = True
        Did_gambit = False
        Repeat_gambit = False
        import scipy.special as scp
        def sheet_of_B(z, sheet_center=0.5, sheet_width=0.1, **kwargs):
            def match_Phi(z, f=scp.erf, center=0.5, width=0.025):
                return 1/2*(1-f((z-center)/width))
            return (1-match_Phi(z, center=sheet_center-sheet_width/2, **kwargs))*(match_Phi(z, center=sheet_center+sheet_width/2, **kwargs))
        
    try:
        start_time = time.time()
        while solver.ok:

            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            if threeD and solver.iteration % Hermitian_cadence == 0:
                for field in solver.state.fields:
                    field.require_grid_space()

            # update lists
            if solver.iteration % report_cadence == 0:
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration, solver.sim_time, solver.sim_time/atmosphere.buoyancy_time, dt)
                log_string += 'Re: {:8.3e}/{:8.3e}'.format(flow.grid_average('Re'), flow.max('Re'))
                if MHD:
                     log_string += ', divB: {:8.3e}/{:8.3e}'.format(flow.grid_average('divB'), flow.max('divB'))
                     #log_string += ', divA: {:8.3e}/{:8.3e}'.format(flow.grid_average('divA'), flow.max('divA'))
                logger.info(log_string)

            if MHD and Tobias_gambit:
                if solver.sim_time/atmosphere.buoyancy_time >= 30 and not Did_gambit:
                    logger.info("Enacting Tobias Gambit")
                    Bx = solver.state['Bx']
                    Bx.set_scales(1, keep_data=True)
                    B0 = np.sqrt(atmosphere.epsilon)
                    Bx['g'] = Bx['g'] + B0*sheet_of_B(atmosphere.z, sheet_center=atmosphere.Lz/2, sheet_width=atmosphere.Lz*0.1)
                    Bx.antidifferentiate('z',('left',0), out=Ay)
                    Ay['g'] *= -1
                    Did_gambit = True
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        end_time = time.time()

        # Print statistics
        elapsed_time = end_time - start_time
        elapsed_sim_time = solver.sim_time
        N_iterations = solver.iteration 
        logger.info('main loop time: {:e}'.format(elapsed_time))
        logger.info('Iterations: {:d}'.format(N_iterations))
        logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
        if N_iterations > 0:
            logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))
        
        logger.info('beginning join operation')
        if do_checkpointing:
            logger.info(data_dir+'/checkpoint/')
            post.merge_analysis(data_dir+'/checkpoint/')

        for task in analysis_tasks:
            logger.info(analysis_tasks[task].base_path)
            post.merge_analysis(analysis_tasks[task].base_path)

        if (atmosphere.domain.distributor.rank==0):

            logger.info('main loop time: {:e}'.format(elapsed_time))
            logger.info('Iterations: {:d}'.format(N_iterations))
            logger.info('iter/sec: {:g}'.format(N_iterations/(elapsed_time)))
            if N_iterations > 0:
                logger.info('Average timestep: {:e}'.format(elapsed_sim_time / N_iterations))
 
            N_TOTAL_CPU = atmosphere.domain.distributor.comm_cart.size

            # Print statistics
            print('-' * 40)
            total_time = end_time-initial_time
            main_loop_time = end_time - start_time
            startup_time = start_time-initial_time
            n_steps = solver.iteration-1
            print('  startup time:', startup_time)
            print('main loop time:', main_loop_time)
            print('    total time:', total_time)
            if N_iterations > 0:
                print('    iterations:', solver.iteration)
                print(' loop sec/iter:', main_loop_time/solver.iteration)
                print('    average dt:', solver.sim_time / n_steps)
                print("          N_cores, Nx, Nz, startup     main loop,   main loop/iter, main loop/iter/grid, n_cores*main loop/iter/grid")
                print('scaling:',
                    ' {:d} {:d} {:d}'.format(N_TOTAL_CPU,nx,nz),
                    ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                                    main_loop_time, 
                                                                    main_loop_time/n_steps, 
                                                                    main_loop_time/n_steps/(nx*nz), 
                                                                    N_TOTAL_CPU*main_loop_time/n_steps/(nx*nz)))
            print('-' * 40)

if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    
    import sys
    # save data in directory named after script
    data_dir = sys.argv[0].split('.py')[0]
    if args['--3D']:
        data_dir +='_3D'
    if args['--fixed_T']:
        data_dir +='_fixed'
    if args['--fixed_Tz']:
        data_dir +='_flux'
    data_dir += "_nrhocz{}_Ra{}".format(args['--n_rho_cz'], args['--Rayleigh'])
    if args['--aspect']:
        data_dir+="_a{}".format(args['--aspect'])
    if args['--MHD']:
        data_dir+= '_MHD'
    if args['--label'] is not None:
        data_dir += "_{}".format(args['--label'])
    data_dir += '/'
    logger.info("saving run in: {}".format(data_dir))
    
    nx =  args['--nx']
    if nx is not None:
        nx = int(nx)
    ny =  args['--ny']
    if ny is not None:
        ny = int(ny)
    nz = args['--nz']
    if nz is None:
        nz = args['--nz_cz']
    if nz is not None:
        nz = int(nz)

    mesh = args['--mesh']
    if mesh is not None:
        mesh = mesh.split(',')
        mesh = [int(mesh[0]), int(mesh[1])]

    FC_polytrope(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      MagneticPrandtl=float(args['--MagneticPrandtl']),
                      aspect_ratio=float(args['--aspect']),
                      threeD=args['--3D'],
                      mesh=mesh,
                      nz=nz,
                      nx=nx,
                      ny=ny,
                      fixed_T=args['--fixed_T'],
                      fixed_Tz=args['--fixed_Tz'],                      
                      MHD=args['--MHD'],
                      restart=(args['--restart']),
                      rk222=args['--rk222'],
                      n_rho_cz=float(args['--n_rho_cz']),
                      data_dir=data_dir,
                      run_time=float(args['--run_time']))
