from functools import partial
import h5py
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy
import time

import bacco
import bacco.probabilistic_bias as pb


def main():
    
    overwrite = True
    n_threads_mp = 1
    n_threads_bacco = 8
    ndens_target = 0.003
    tag_bpfit = ''
    #redshift = 0
    #dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    #idxs_sam = [idx_sam for idx_sam in np.arange(0, 1000) \
    #            if os.path.isfile(f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5')]
    
    fn_idxs = '../data/idxs_camelssam_in_emu_bounds.dat'
    idxs_sam_inbounds = np.loadtxt(fn_idxs, dtype=int)
    print(f'{len(idxs_sam_inbounds)} of SAMs have cosmo params in bounds') 
    # idx_sam = 12 not working!! first in idxs_sam_inbounds, avoid w this line:
    idxs_sam_inbounds = idxs_sam_inbounds[1:]
    print(idxs_sam_inbounds)
    
    #TESTING SINGLE
    idxs_sam_inbounds = idxs_sam_inbounds[:5]
    
    fn_params = '../data/params_CAMELS-SAM.dat'
    df_params = pd.read_csv(fn_params, index_col='idx_LH')
    
    dir_bp = f'../data/probabilistic_bias_params/bias_params{tag_bpfit}'
    Path(dir_bp).mkdir(parents=True, exist_ok=True)

    bacco.configuration.update({'number_of_threads': n_threads_bacco})

    fit_prob_bias_params_loop(idxs_sam_inbounds, df_params, dir_bp, tag_bpfit, ndens_target, 
                              n_threads_mp=n_threads_mp, n_threads_bacco=n_threads_bacco, 
                              overwrite=overwrite)



def fit_prob_bias_params_loop(idxs_sam, df_params, dir_bp, tag_bpfit, ndens_target, 
                              n_threads_mp=2, n_threads_bacco=4, overwrite=False):
    
    #vol_Mpc = (100/cosmo_params['hubble'])**3 
    vol_hMpc = 100**3

    start = time.time()
    if n_threads_mp>1:
        pool = mp.Pool(processes=n_threads_mp)
        print("Starting multiprocessing pool")
        outputs = pool.map(partial(fit_prob_bias_params_single,
                                   df_params=df_params, 
                                   dir_bp=dir_bp, tag_bpfit=tag_bpfit, ndens_target=ndens_target,
                                   vol=vol_hMpc, n_threads_bacco=n_threads_bacco, overwrite=overwrite), idxs_sam)
        print("Done!")
    else:
        print("Starting serial loop")
        outputs = []
        for idx_sam in idxs_sam:
            output = fit_prob_bias_params_single(idx_sam, df_params=df_params, 
                                     dir_bp=dir_bp, tag_bpfit=tag_bpfit, ndens_target=ndens_target,
                                     vol=vol_hMpc, n_threads_bacco=n_threads_bacco, overwrite=overwrite)
            outputs.append(output)
    end = time.time()
    outputs = np.array(outputs)
    n_success = np.sum(outputs==0)
    print(f"Took {(end-start)/60} min to fit {n_success} bias param sets with N={n_threads_mp} \
            mp threads, {n_threads_bacco} bacco threads")


def fit_prob_bias_params_single(idx_sam, df_params=None, dir_bp=None, tag_bpfit=None,
                    ndens_target=None, vol=None, n_threads_bacco=4,overwrite=False):
    
    box_size = 100.
    ngrid = 640 #TODO read in?
    LPT_order = 2
    seed_lpt = idx_sam + 5000 # from camels data gen
    damping_scale=0.2 #TODO what to use?
    
    bacco.configuration.update({'scaling' : {'disp_ngrid' : 640}})
    bacco.configuration.update({'scaling' : {'LPT_order' : LPT_order}})
    bacco.configuration.update({'number_of_threads': n_threads_bacco})

    fn_bp = f'{dir_bp}/bias_params_LH_{idx_sam}.npy'
    if os.path.isfile(fn_bp) and not overwrite:
        print(f"[SAM LH {idx_sam}] Prob bias param file {fn_bp} already exists and overwrite={overwrite}! Moving on")
        return 1
    
    redshift = 0
    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    fn_dat = f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5'
    if not os.path.isfile(fn_dat):
        raise ValueError(f"[SAM LH {idx_sam}] Data file {fn_dat} does not exist!")

    cosmo = setup_cosmo(df_params.loc[idx_sam])

    # CREATE A ZA SIMULATION
    print("Generating LPT sim")
    s = time.time()
    sim = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=ngrid, Seed=seed_lpt,
                                                        FixedInitialAmplitude=False,InitialPhase=0, 
                                                        expfactor=1.0, LPT_order=LPT_order, order_by_order=None,
                                                        phase_type=1, ngenic_phases=True, return_disp=False, 
                                                        sphere_mode=0)
    e = time.time()
    print(f"LPT sim generated in {(e-s)/60} min")

    print("Getting qdata and tracer values")
    s = time.time()
    # Define what variables we consider, "J2" means density and "J4" means Laplacian
    spatial_order = 2
    variables = ("J2", "J2=2")
    pbm = pb.ProbabilisticBiasManager(sim, variables=variables, damping_scale=damping_scale, ngrid=ngrid) 
    # Define what bias parameters we want to measure
    # "J2" corresponds to b1, "J22" to b2, "J24" to bdeltaL, "J4" to bL, "J44" to bL**2
    #terms = ("J2", "J22", "J24", "J4", "J44")
    terms = ("J2", "J22", "J2=2")
    #param_names = ['b1', 'b2', '2bs2']
    model_expansion = pbm.setup_bias_model(pb.TensorBiasND, terms=terms, spatial_order=spatial_order)

    print('load tracer data', flush=True)
    pos_arr_hMpc, vel_arr = load_tracer_data(fn_dat, ndens_target, vol)
    print("create qdata", flush=True)
    # pos_arr_hMpc = pos_arr_hMpc[:10]
    # vel_arr = vel_arr[:10]
    # halo_id_arr = halo_id_arr[:10]
    
    # id is a dummy, barely affects output
    id_arr = np.ones(len(pos_arr_hMpc), dtype=int)
    qdata = create_qdata_custom(sim, pos_arr_hMpc, vel_arr, id_arr, ngrid,
                                number_of_threads=n_threads_bacco)

    print('interpolate to get tracer values', flush=True)
    tracer_q = qdata['pos']
    e = time.time()
    print(f"qdata and tracer values gotten in {(e-s)/60} min", flush=True)
    
    
    print("Fitting bias")
    b, bcov = pbm.fit_bias(model=model_expansion, tracer_q=tracer_q,
                           #tracer_value=tracer_value, 
                           error="jack4")

    print('bias vec:', b)
    
    bias_params_fit_dict = dict(zip(terms, b))
    bias_data = [bias_params_fit_dict, bcov]
    np.save(fn_bp, bias_data)
    print(f"Fit for SAM {idx_sam} complete. Saved bias params to {fn_bp}", flush=True)
    return 0

    
    
def load_tracer_data(fn_dat, ndens_target, vol_hMpc):
        # data description: https://camels-sam.readthedocs.io/en/main/dataproducts.html
    # (note names don't match, not sure why - col names here: https://camels-sam.readthedocs.io/en/main/openSAM.html)
    vol_hMpc = 100**3 # units Mpc/h!! 
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