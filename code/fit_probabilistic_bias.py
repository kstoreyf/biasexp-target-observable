from functools import partial
import h5py
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import scipy
import time

import bacco
import bacco.probabilistic_bias as pb

import utils


def main():
    
    overwrite = False
    n_threads_mp = 6
    n_threads_bacco = 2
    ndens_target = 0.003
    tag_bpfit = '_wsigma0'
    #tag_bpfit = '_wsigma0_tracerqeul'
    #redshift = 0
    #dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    #idxs_sam = [idx_sam for idx_sam in np.arange(0, 1000) \
    #            if os.path.isfile(f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5')]
    
    #fn_idxs = '../data/idxs_camelssam_in_emu_bounds.dat'
    #idxs_sam = np.loadtxt(fn_idxs, dtype=int)
    #print(f'{len(idxs_sam)} of SAMs have cosmo params in bounds') 
    
    idxs_sam = np.arange(1000)
    #idxs_sam = np.array([1,2])
    # idx_sam = 12 not working (TODO understand why)!! avoid w this line:
    idxs_sam = np.delete(idxs_sam, np.argwhere(idxs_sam==12))
    print(idxs_sam)
    
    #TESTING SINGLE
    #idxs_sam_inbounds = idxs_sam_inbounds[:5]
    
    fn_params = '../data/params_CAMELS-SAM.dat'
    df_params = pd.read_csv(fn_params, index_col='idx_LH')
    
    dir_bp = f'../data/probabilistic_bias_params/bias_params{tag_bpfit}'
    Path(dir_bp).mkdir(parents=True, exist_ok=True)

    bacco.configuration.update({'number_of_threads': n_threads_bacco})

    fit_prob_bias_params_loop(idxs_sam, df_params, dir_bp, tag_bpfit, ndens_target, 
                              n_threads_mp=n_threads_mp, n_threads_bacco=n_threads_bacco, 
                              overwrite=overwrite)



def fit_prob_bias_params_loop(idxs_sam, df_params, dir_bp, tag_bpfit, ndens_target, 
                              n_threads_mp=2, n_threads_bacco=4, overwrite=False):
    
    #vol_Mpc = (100/cosmo_params['hubble'])**3 
    box_size = 100.0
    
    pbm = setup_pbm()

    start = time.time()
    if n_threads_mp>1:
        pool = mp.Pool(processes=n_threads_mp)
        print("Starting multiprocessing pool")
        outputs = pool.map(partial(fit_prob_bias_params_single,
                                   df_params=df_params, 
                                   dir_bp=dir_bp, tag_bpfit=tag_bpfit, ndens_target=ndens_target,
                                   box_size=box_size, n_threads_bacco=n_threads_bacco, overwrite=overwrite), idxs_sam)
        print("Done!")
    else:
        print("Starting serial loop")
        outputs = []
        for idx_sam in idxs_sam:
            output = fit_prob_bias_params_single(idx_sam, df_params=df_params, 
                                     dir_bp=dir_bp, tag_bpfit=tag_bpfit, ndens_target=ndens_target,
                                     box_size=box_size, n_threads_bacco=n_threads_bacco, overwrite=overwrite)
            outputs.append(output)
    end = time.time()
    outputs = np.array(outputs)
    n_success = np.sum(outputs==0)
    print(f"Took {(end-start)/60} min to fit {n_success} bias param sets with N={n_threads_mp} \
            mp threads, {n_threads_bacco} bacco threads")



def setup_sim(sim_name='millennium', idx_sam=None, n_threads_bacco=8,
              df_params=None):
    
    s = time.time()

    if sim_name=='millennium':
        n_grid = 512 #??
        # believe LPT order is 1 (ZA); see Springel 2005 (https://arxiv.org/pdf/astro-ph/0505010)
        # which cites Heitmann 2004 (https://arxiv.org/pdf/astro-ph/0411795, S5.1)
        LPT_order = 1 
        # TODO changing LPT_order doesn't seem to be doing anything, CHECK
        seed_millennium = 100672
        cosmo, box_size = utils.setup_cosmo('millennium_planck', return_box_size=True)
        sim = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid, 
                                                Seed=seed_millennium,
                                                FixedInitialAmplitude=False, InitialPhase=0, 
                                                expfactor=1.0, LPT_order=LPT_order, order_by_order=None,
                                                phase_type=None,
                                                ngenic_phases=False, 
                                                millennium_ics=True,
                                                return_disp=False, 
                                                sphere_mode=0)
        
    elif sim_name=='camelssam':
        
        if idx_sam is None:
            raise ValueError("Must specify idx_sam")
        
        ngrid = 640 #TODO read in?
        box_size = 100.0
        LPT_order = 2
        seed_lpt = idx_sam + 5000 # from camels data gen

        if df_params is None:
            raise ValueError("Must specify df_params for CAMELS-SAM")
        cosmo = setup_cosmo(df_params.loc[idx_sam])

        bacco.configuration.update({'scaling' : {'disp_ngrid' : 640}})
        bacco.configuration.update({'scaling' : {'LPT_order' : LPT_order}})
        bacco.configuration.update({'number_of_threads': n_threads_bacco})
        
        # CREATE A ZA SIMULATION
        print("Generating LPT sim")
        sim = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=ngrid, Seed=seed_lpt,
                                                            FixedInitialAmplitude=False,InitialPhase=0, 
                                                            expfactor=1.0, LPT_order=LPT_order, order_by_order=None,
                                                            phase_type=1, ngenic_phases=True, return_disp=False, 
                                                            sphere_mode=0)

    else:
        raise ValueError(f"Invalid sim_name {sim_name}")
    
    e = time.time()
    print(f"LPT sim generated in {(e-s)/60} min")

    return sim


def setup_pbm(sim, damping_scale=0.2):
    
    s = time.time()
    ngrid = sim.Nmesh #TODO check if this is fine??
    print(ngrid)
    # Define what variables we consider, "J2" means density and "J4" means Laplacian
    spatial_order = 2
    variables = ("J2", "J2=2")
    print("damping scale =", damping_scale)
    pbm = pb.ProbabilisticBiasManager(sim, variables=variables, damping_scale=damping_scale, ngrid=ngrid) 
    # Define what bias parameters we want to measure
    # "J2" corresponds to b1, "J22" to b2, "J24" to bdeltaL, "J4" to bL, "J44" to bL**2
    #terms = ("J2", "J22", "J24", "J4", "J44")
    terms = ("J2", "J22", "J2=2")
    # this becomes self.bias_model
    pbm.setup_bias_model(pb.TensorBiasND, terms=terms, spatial_order=spatial_order)
    e = time.time()
    print(f"PB setup in {(e-s)/60} min")

    return pbm



def fit_prob_bias_params_single(sim, pbm, tag, idx_sam, df_params=None, dir_bp=None, tag_bpfit=None,
                    ndens_target=None, box_size=None, n_threads_bacco=4,overwrite=False):
    
    redshift = 0
    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    fn_dat = f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5'
    if not os.path.isfile(fn_dat):
        raise ValueError(f"[SAM LH {idx_sam}] Data file {fn_dat} does not exist!")

    fn_bp = f'{dir_bp}/bias_params_LH_{idx_sam}.npy'
    if os.path.isfile(fn_bp) and not overwrite:
        print(f"[SAM LH {idx_sam}] Prob bias param file {fn_bp} already exists and overwrite={overwrite}! Moving on")
        return 1
    
    s = time.time()

    print("Getting qdata and tracer values")

    print('load tracer data', flush=True)
    pos_arr_hMpc, vel_arr = load_tracer_data(fn_dat, ndens_target, box_size)
    print("create qdata", flush=True)
    
    # tinytest purposes
    # pos_arr_hMpc = pos_arr_hMpc[:10]
    # vel_arr = vel_arr[:10]
    # halo_id_arr = halo_id_arr[:10]
    
    ngrid = sim.Nmesh #TODO check if this is fine??

    # id is a dummy, barely affects output
    id_arr = np.ones(len(pos_arr_hMpc), dtype=int)
    qdata = create_qdata_custom(sim, pos_arr_hMpc, vel_arr, id_arr, ngrid,
                               number_of_threads=n_threads_bacco)

    tracer_q = qdata['pos']
    #print("EULERIAN POSITIONS AS TEST")
    #tracer_q = pos_arr_hMpc #EULERIAN POSITIONS AS TEST
    e = time.time()
    print(f"qdata and tracer values gotten in {(e-s)/60} min", flush=True)
    
    
    print("Fitting bias")
    b, bcov = pbm.fit_bias(model=pbm.bias_model, tracer_q=tracer_q,
                           #tracer_value=tracer_value, 
                           error="jack4")
    bias_per_object = pbm.bias_model.bias_per_object(pbm.tr_value)
    print('bias vec:', b)
    
    print("Get sigma0")
    fields_sim_damped = sim.get_linear_field(ngrid=ngrid, damping_scale=pbm.damping_scale)
    overdensity_bacco_damped = fields_sim_damped[0]
    sigma_0_damped = np.sqrt(np.mean(overdensity_bacco_damped**2))
    print("sigma0:", sigma_0_damped)

    bias_params_fit_dict = dict(zip(pbm.terms, b))
    #bias_data = [bias_params_fit_dict, bcov, bias_per_object, sigma_0_damped]
    bias_data = {'bias_param_dict': bias_params_fit_dict, 
                 'bias_param_cov': bcov, 
                 'bias_per_object': bias_per_object, 
                 'sigma_0_damped': sigma_0_damped}
    np.save(fn_bp, bias_data)
    print(f"Fit for SAM {idx_sam} complete. Saved bias params to {fn_bp}", flush=True)
    return 0


def get_volume_Mpc(box_size, h, fn_dat=None, n_trees=None):
    assert fn_dat is not None or n_trees is not None, "Need either fn_dat or n_trees"
    TotTreeFiles = 512
    if fn_dat is not None:
        matches = re.search(r"tree(\d+)-(\d+)", fn_dat)
        ft, lt = int(matches.group(1)), int(matches.group(2))
        #numbers = re.findall(r'\d+', tag_trees)
        #ft, lt = int(numbers[0]), int(numbers[1])
        TreeFilesUsed_thisfile = lt - ft + 1
    else:
        TreeFilesUsed_thisfile = n_trees
    vol_Mpch_thisfile = box_size**3 * TreeFilesUsed_thisfile / TotTreeFiles
    vol_Mpc_thisfile = vol_Mpch_thisfile / h**3 # X Mpc/h * (h/0.7) = X/0.7 Mpc
    return vol_Mpc_thisfile
    
    
def load_tracer_data_lgalaxies(ndens_target, box_size, h,
                               fn_dat=None, gals=None,
                               n_trees=None,
                               return_masses=False):

    assert fn_dat is not None or gals is not None, "Need either fn_dat or gals"
    if gals is not None:
        assert n_trees is not None, "If using gals, need n_trees"
    # from main_lgals.py:
    # Volume = (BoxSideLength**3.0) * TreeFilesUsed / TotTreeFiles
    # TotTreeFiles = 512
    # vol_hMpc = box_size**3 * TreeFilesUsed / TotTreeFiles
    vol_Mpc_thisfile = get_volume_Mpc(box_size, h, fn_dat=fn_dat, n_trees=n_trees)
    n_target = int(ndens_target * vol_Mpc_thisfile)
    
    if fn_dat is not None:
        gals = np.load(fn_dat)
    print("Total number of gals initially:", len(gals))
    
    log_mstar = np.log10(gals['StellarMass'])
    i_target = np.argsort(log_mstar)[::-1][:n_target] # order by mstar and take largest to smallest to get desired ndens
    log_mstar = log_mstar[i_target]

    # Pos, 1/h Mpc , Comoving galaxy/subhalo position
    # (from ./input/hdf5_field_props.txt)
    pos_arr = gals['Pos']
    pos_arr_hMpc = pos_arr*h
    pos_arr_hMpc = pos_arr_hMpc[i_target]

    # velocity in km/s
    vel_arr = gals['Vel']
    vel_arr = vel_arr[i_target]
    
    if return_masses:
        log_mvir = np.log10(gals['Mvir'])
        log_mvir = log_mvir[i_target]
        return pos_arr_hMpc, vel_arr, log_mstar, log_mvir
    else:
        return pos_arr_hMpc, vel_arr
    
    
    
def load_tracer_data_camelssam(fn_dat, ndens_target, box_size):
        # data description: https://camels-sam.readthedocs.io/en/main/dataproducts.html
    # (note names don't match, not sure why - col names here: https://camels-sam.readthedocs.io/en/main/openSAM.html)
    vol_hMpc = box_size**3 # units Mpc/h!! 
    n_target = int(ndens_target * vol_hMpc)
    log_mass_shift = 9

    with h5py.File(fn_dat, 'r') as f:
        print(f.keys())
        mstar_raw = np.array(f['mstar'])
        i_target = np.argsort(mstar_raw)[::-1][:n_target] # order by mstar and take largest to smallest to get desired ndens
        log_mstar = np.log10(mstar_raw) + log_mass_shift
        log_mstar = log_mstar[i_target]
        log_mhalo = np.log10(np.array(f['mhalo'])) + log_mass_shift
        log_mhalo = log_mhalo[i_target]

        # position in Mpc (comoving)
        x_arr, y_arr, z_arr = f['x_position'], f['y_position'], f['z_position']
        pos_arr = np.array([x_arr, y_arr, z_arr]).T
        pos_arr = pos_arr[i_target]

        # velocity in km/s
        vx_arr, vy_arr, vz_arr = f['vx'], f['vy'], f['vz']
        vel_arr = np.array([vx_arr, vy_arr, vz_arr]).T
        vel_arr = vel_arr[i_target]
        
        halo_id_arr = np.array(f['halo_index'], dtype=int)
        halo_id_arr = halo_id_arr[i_target]

    # to hMpc to match volume    
    h = 0.6711
    pos_arr_hMpc = pos_arr*h

    # don't know why negative or >1000!! for now just cut, only few per sim - but TODO ask Lucia
    i_oob = np.any(pos_arr_hMpc<0,axis=1) | np.any(pos_arr_hMpc>box_size,axis=1)
    print(f'Found {np.sum(i_oob)} positions with negative values! cutting')
    pos_arr_hMpc = pos_arr_hMpc[~i_oob,:]
    vel_arr = vel_arr[~i_oob,:]
    
    return pos_arr_hMpc, vel_arr


# adapted from bacco.simulation.create_qdata()
# https://bitbucket.org/rangulo/baccogit/src/5766697248ceb5d0554a4c78c067dd6309f008ee/bacco/simulation.py#lines-5850
def create_qdata_custom(sim, pos, vel, ids, ngrid, qdata_tolerance=0.01, 
                        number_of_threads=1, verbose=True): 

    pos = np.float32(pos)
    vel = np.float32(vel)

    vel_factor = sim.Cosmology.vel_factor(sim.Cosmology.expfactor)

    nbodies = np.size(pos[:, 0])

    _qdata = {'pos': np.zeros((nbodies, 3), dtype=np.float32),
            'vel': np.zeros((nbodies, 3), dtype=np.float32)}
    bias = np.repeat(np.float32(1.), nbodies)
    print("qdata creation set up, finding lagrangian coordinates")

    print('nbodies', nbodies)
    istart = 0
    nchunks = 8 #20
    for i in range(nchunks):
        print("chunk", i, flush=True)
        iend = istart + int(nbodies / nchunks)
        if i == (nchunks - 1):
            iend = nbodies
        print(istart, iend)

        # i think what was sim.header['Nsample'] is ngrid, bc 
        # https://bitbucket.org/rangulo/baccogit/src/5766697248ceb5d0554a4c78c067dd6309f008ee/bacco/utils.py?at=master#lines-1082
        _qpos, _qvel = bacco.lss_scaler.lss.find_lagrangian_coordinates(
            pos[istart:iend, :], vel[istart:iend, :],
            ids[istart:iend], bias[istart:iend],
            qdata_tolerance, ngrid,
            sim.disp_field, sim.header['BoxSize'],
            vel_factor, number_of_threads,
            verbose)
        print('scaler done', flush=True)
        _qdata['pos'][istart:iend, :] = _qpos
        _qdata['vel'][istart:iend, :] = _qvel
        istart = iend
    print("Qdata complete!")
        
    return _qdata


def setup_cosmo(params):
    print("Setting up cosmology")
    assert 'Omega_m' in params and 'sigma_8' in params, "params dict must contain Omega_m and sigma_8!"
    # CAMELS-SAM sim params, fixed
    Ob = 0.049
    hubble = 0.6711
    ns = 0.9624
    cosmopars = dict(
            omega_cdm=params['Omega_m']-Ob,
            omega_baryon=Ob, 
            hubble=hubble, 
            ns=ns, 
            sigma8=params['sigma_8'],
            tau=0.0561,
            A_s=None,
            neutrino_mass=0.,
            w0=-1,
            wa=0,
        )
        
    cosmo = bacco.Cosmology(**cosmopars)
    cosmo.set_expfactor(1.0)
    return cosmo


if __name__ == "__main__":
    main()
