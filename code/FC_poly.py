"""
Dedalus script for 2D or 3D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options] 

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho_cz=<n_rho_cz>                Density scale heights across unstable layer [default: 3.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 1e-4]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]
    
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz
    --ny=<ny>                            Horizontal y (Fourier) resolution; if not set, ny=nx (3D only) 
    --3D                                 Do 3D run
    --mesh=<mesh>                        Processor mesh if distributing 3D run in 2D 

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time>           Run time, in buoyancy times

    --fixed_T                            Fixed Temperature boundary conditions (top and bottom)
    --fixed_flux                         Fixed flux boundary conditions (top and bottom)
    --const_nu                           If flagged, use constant nu 
    --const_chi                          If flagged, use constant chi 

    --restart=<restart_file>             Restart from checkpoint
    --start_new_files                    Start new files while checkpointing

    --rk222                              Use RK222 as timestepper
    --safety_factor=<safety_factor>      Determines CFL Danger.  Higher=Faster [default: 0.2]
    --split_diffusivities                If True, split the chi and nu between LHS and RHS to lower bandwidth
    
    --root_dir=<root_dir>                Root directory to save data dir in [default: ./]
    --label=<label>                      Additional label for run output directory
    --out_cadence=<out_cadence>          The fraction of a buoyancy time to output data at [default: 0.1]
    --no_coeffs                          If flagged, coeffs will not be output
"""
import logging
logger = logging.getLogger(__name__)


import dedalus.public as de
from dedalus.tools  import post
from dedalus.extras import flow_tools
import numpy as np

try:
    from dedalus.extras.checkpointing import Checkpoint
    do_checkpointing = True
    checkpoint_min   = 60
except:
    print('not importing checkpointing')
    do_checkpointing = False

def FC_polytrope(  Rayleigh=1e4, Prandtl=1, aspect_ratio=4,\
                        nz=128, nx=None, ny=None, threeD=False, mesh=None,\
			            n_rho_cz=3.5, epsilon=1e-4, run_time=23.5, \
                        fixed_T=False, fixed_flux=False, const_mu=True, const_kappa=True,\
                        restart=None, start_new_files=False, \
                        rk222=False, safety_factor=0.2, run_time_buoyancies=None, \
                        data_dir='./', out_cadence=0.1, no_coeffs=False,
                        split_diffusivities=False):
    import time
    import equations
    import os
    import sys
    
    initial_time = time.time()

    logger.info("Starting Dedalus script {:s}".format(sys.argv[0]))

    if nx is None:
        nx = int(np.round(nz*aspect_ratio))
    if threeD and ny is None:
        ny = nx
    
    if threeD:
        atmosphere = equations.FC_polytrope_3d(nx=nx, ny=ny, nz=nz, mesh=mesh, constant_kappa=const_kappa, constant_mu=const_mu,\
                                        epsilon=epsilon, n_rho_cz=n_rho_cz, aspect_ratio=aspect_ratio,\
                                        fig_dir='./FC_poly_atmosphere/')
    else:
        atmosphere = equations.FC_polytrope_2d(nx=nx, nz=nz, constant_kappa=const_kappa, constant_mu=const_mu,\
                                        epsilon=epsilon, n_rho_cz=n_rho_cz, aspect_ratio=aspect_ratio,\
                                        fig_dir='./FC_poly_atmosphere/')
    if epsilon < 1e-4:
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, ncc_cutoff=1e-14, split_diffusivities=split_diffusivities)
    elif epsilon > 1e-1:
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, ncc_cutoff=1e-6, split_diffusivities=split_diffusivities)
    else:
        atmosphere.set_IVP_problem(Rayleigh, Prandtl, ncc_cutoff=1e-10, split_diffusivities=split_diffusivities)

    if fixed_T:
        atmosphere.set_BC(fixed_temperature=True, stress_free=True)
    elif fixed_flux:
        atmosphere.set_BC(fixed_flux=True, stress_free=True)
    else:
        atmosphere.set_BC(mixed_flux_temperature=True, stress_free=True)

    problem = atmosphere.get_problem()

    if atmosphere.domain.distributor.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))

    if rk222:
        logger.info("timestepping using RK222")
        ts = de.timesteppers.RK222
        cfl_safety_factor = safety_factor*2
    else:
        logger.info("timestepping using RK443")
        ts = de.timesteppers.RK443
        cfl_safety_factor = safety_factor*4

    # Build solver
    solver = problem.build_solver(ts)

    #Check atmosphere
    logger.info("thermal_time = {:g}, top_thermal_time = {:g}".format(atmosphere.thermal_time,\
                                                                    atmosphere.top_thermal_time))
    logger.info("full atm HS check")
    atmosphere.check_atmosphere(make_plots = False, rho=atmosphere.get_full_rho(solver), T=atmosphere.get_full_T(solver))

    #Set up timestep defaults
    max_dt = atmosphere.buoyancy_time*0.05
    dt = max_dt/5
    if epsilon < 1e-5:
        max_dt = atmosphere.buoyancy_time*0.05
        dt     = max_dt

  
    if restart is None or start_new_files or not do_checkpointing:
        slices_count, slices_set        = 1,1
        profiles_count, profiles_set    = 1,1
        scalar_count, scalar_set        = 1,1
        coeffs_count, coeffs_set        = 1,1
        chk_write = chk_set = 1
    if do_checkpointing:
        logger.info('checkpointing in {}'.format(data_dir))

        #Find all of the directories we don't want to checkpoint in
        import glob
        good_dirs = ['slices', 'profiles', 'scalar', 'coeffs', 'checkpoint']
        dirs = glob.glob('{:s}/*/'.format(data_dir))
        found_dirs = [s_dir.split(data_dir)[-1].split('/')[0] for s_dir in dirs]
        excluded_dirs = []
        for found_dir in found_dirs:
            if found_dir not in good_dirs: excluded_dirs.append(found_dir)

        #Checkpoint
        try:
            checkpoint = Checkpoint(data_dir, excluded_dirs=excluded_dirs)
        except:
            checkpoint = Checkpoint(data_dir)

        if restart is None:
            atmosphere.set_IC(solver)
        else:
            logger.info("restarting from {}".format(restart))
            chk_write, chk_set, dt = checkpoint.restart(restart, solver)
            if not start_new_files:
                counts, sets = checkpoint.find_output_counts()
                #All of the +1s make it so that we make a new file rather than overwriting the previous.
                slices_count, slices_set            = counts['slices']+1,sets['slices']+1
                profiles_count, profiles_set      = counts['profiles']+1,sets['profiles']+1
                scalar_count, scalar_set            = counts['scalar']+1,sets['scalar']+1
                try: #Allows for runs without coeffs
                    coeffs_count, coeffs_set = counts['coeffs']+1, sets['coeffs']+1
                except:
                    coeffs_count, coeffs_set = 1, 1
                chk_write += 1
                chk_set   += 1
            else:
                chk_write = chk_set = 1
        checkpoint.set_checkpoint(solver, wall_dt=checkpoint_min*60, write_num=chk_write, set_num=chk_set)
    else:
        atmosphere.set_IC(solver, shear=shear, shear_value=shear_val)


    
    if run_time_buoyancies != None:
        solver.stop_sim_time    = solver.sim_time + run_time_buoyancies*atmosphere.buoyancy_time
    else:
        solver.stop_sim_time    = 100*atmosphere.thermal_time
    
    solver.stop_iteration   = np.inf
    solver.stop_wall_time   = run_time*3600
    report_cadence = 1
    output_time_cadence = out_cadence*atmosphere.buoyancy_time
    if threeD:
        Hermitian_cadence = 10
    else:
        Hermitian_cadence = 100
    
    logger.info("stopping after {:g} time units".format(solver.stop_sim_time))
    logger.info("output cadence = {:g}".format(output_time_cadence))
    
    if no_coeffs:
        coeffs_output=False
    else:
        coeffs_output=True
    analysis_tasks = atmosphere.initialize_output(solver, data_dir, sim_dt=output_time_cadence, coeffs_output=coeffs_output,\
                                slices=[slices_count, slices_set], profiles=[profiles_count, profiles_set], scalar=[scalar_count, scalar_set],\
                                coeffs=[coeffs_count, coeffs_set])
    
    cfl_cadence = 1
    cfl_threshold=0.1
    CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=cfl_cadence, safety=cfl_safety_factor,
                         max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=cfl_threshold)
    if threeD:
        CFL.add_velocities(('u', 'v', 'w'))
    else:
        CFL.add_velocities(('u', 'w'))

    # Flow properties
    flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
    flow.add_property("Re_rms", name='Re')
    flow.add_property("Pe_rms", name='Pe')
    flow.add_property("Nusselt", name='Nusselt')
    
    start_iter=solver.iteration
    start_sim_time = solver.sim_time

    try:
        start_time = time.time()
        logger.info('starting main loop')
        while solver.ok:

            dt = CFL.compute_dt()
            # advance
            solver.step(dt)

            if threeD and solver.iteration % Hermitian_cadence == 0:
                for field in solver.state.fields:
                    field.require_grid_space()

            # update lists
            if solver.iteration % report_cadence == 0:
                log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:8.3e}), dt: {:8.3e}, '.format(solver.iteration-start_iter, solver.sim_time, (solver.sim_time-start_sim_time)/atmosphere.buoyancy_time, dt)
                log_string += '\n\t\tRe: {:8.5e}/{:8.5e}'.format(flow.grid_average('Re'), flow.max('Re'))
                log_string += '; Pe: {:8.5e}/{:8.5e}'.format(flow.grid_average('Pe'), flow.max('Pe'))
                log_string += '; Nu: {:8.5e}/{:8.5e}'.format(flow.grid_average('Nusselt'), flow.max('Nusselt'))
                logger.info(log_string)
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
            try:
                final_checkpoint = Checkpoint(data_dir, checkpoint_name='final_checkpoint')
                final_checkpoint.set_checkpoint(solver, wall_dt=1, write_num=1, set_num=1)
                solver.step(dt) #clean this up in the future...works for now.
                post.merge_analysis(data_dir+'/final_checkpoint/')
            except:
                print('cannot save final checkpoint')

            logger.info(data_dir+'/checkpoint/')
            post.merge_analysis(data_dir+'/checkpoint/')

        for task in analysis_tasks.keys():
            logger.info(analysis_tasks[task].base_path)
            post.merge_analysis(analysis_tasks[task].base_path)

        if (atmosphere.domain.distributor.rank==0):

            logger.info('main loop time: {:e}'.format(elapsed_time))
            if start_iter > 1:
                logger.info('Iterations (this run): {:d}'.format(N_iterations - start_iter))
                logger.info('Iterations (total): {:d}'.format(N_iterations - start_iter))
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
    #   these lines really are all about setting up the output directory name
    data_dir = args['--root_dir']
    if data_dir[-1] != '/':
        data_dir += '/'
    data_dir += sys.argv[0].split('.py')[0]
    #BCs
    if args['--fixed_T']:
        data_dir += '_fixed'
    elif args['--fixed_flux']:
        data_dir += '_flux'
    #Diffusivities
    if args['--const_nu']:
        data_dir += '_constNu'
    else:
        data_dir += '_constMu'
    if args['--const_chi']:
        data_dir += '_constChi'
    else:
        data_dir += '_constKappa'
    if args['--3D']:
        data_dir +='_3D'
    else:
        data_dir +='_2D'
    #Base atmosphere
    data_dir += "_nrhocz{}_Ra{}_Pr{}_eps{}_a{}".format(args['--n_rho_cz'], args['--Rayleigh'], args['--Prandtl'], args['--epsilon'], args['--aspect'])
    if args['--label'] == None:
        data_dir += '/'
    else:
        data_dir += '_{}/'.format(args['--label'])
    logger.info("saving run in: {}".format(data_dir))
  

    #Timestepper type
    if args['--rk222']:
        rk222=True
    else:
        rk222=False

    #Restarting options
    if args['--start_new_files']:
        start_new_files = True
    else:
        start_new_files = False

    #Resolution
    nx = args['--nx']
    if nx is not None:
        nx = int(nx)
    ny =  args['--ny']
    if ny is not None:
        ny = int(ny)
    nz = int(args['--nz'])

    #Diffusivity flags
    const_mu    = True
    const_kappa = True
    if args['--const_nu']:
        const_mu   = False
    if args['--const_chi']:
        const_kappa = False

    mesh = args['--mesh']
    if mesh is not None:
        mesh = mesh.split(',')
        mesh = [int(mesh[0]), int(mesh[1])]

    run_time_buoy = args['--run_time_buoy']
    if run_time_buoy != None:
        run_time_buoy = float(run_time_buoy)

    FC_polytrope(Rayleigh=float(args['--Rayleigh']),
                      Prandtl=float(args['--Prandtl']),
                      threeD=args['--3D'],
                      mesh=mesh,
                      nx = nx,
                      ny = ny,
                      nz = nz,
                      aspect_ratio=float(args['--aspect']),
                      n_rho_cz=float(args['--n_rho_cz']),
                      epsilon=float(args['--epsilon']),
                      run_time=float(args['--run_time']),
                      run_time_buoyancies=run_time_buoy,
                      fixed_T=args['--fixed_T'],
                      fixed_flux=args['--fixed_flux'],
                      const_mu=const_mu,
                      const_kappa=const_kappa,
                      restart=(args['--restart']),
                      start_new_files=start_new_files,
                      rk222=rk222,
                      safety_factor=float(args['--safety_factor']),
                      out_cadence=float(args['--out_cadence']),
                      data_dir=data_dir,
                      no_coeffs=args['--no_coeffs'],
                      split_diffusivities=args['--split_diffusivities'])
