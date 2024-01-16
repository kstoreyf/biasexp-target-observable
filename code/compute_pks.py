from functools import partial
import h5py
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time

import bacco


def main():

    args_power = setup_bacco()
   
    overwrite = False
    n_threads = 12
    idx_max = 1000
    #idx_max = 1000

    redshift = 0
    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    idxs_sam = [idx_sam for idx_sam in np.arange(0, idx_max) \
                if os.path.isfile(f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5')]

    ndens_target = 0.003 # (Mpc/h)^-3
    tag_pk = f'_n{ndens_target}_hMpc'
    dir_pk = f'../data/pks/pks{tag_pk}'
    Path(dir_pk).mkdir(parents=True, exist_ok=True)
    compute_pks_loop(idxs_sam, ndens_target, tag_pk, args_power, overwrite=overwrite, n_threads=n_threads)


def compute_pks_loop(idxs_sam, ndens_target, tag_pk, args_power, overwrite=False, n_threads=2):
    
    fn_params = '../data/params_CAMELS-SAM.dat'
    df_params = pd.read_csv(fn_params, index_col='idx_LH')

    start = time.time()
    if n_threads>1:
        pool = mp.Pool(processes=n_threads)
        print("Starting multiprocessing pool")
        outputs = pool.map(partial(compute_pk, df_params=df_params, 
                                   ndens_target=ndens_target, tag_pk=tag_pk,
                                   args_power=args_power, overwrite=overwrite), idxs_sam)
        print("Done!")
    else:
        outputs = []
        for idx_sam in idxs_sam:
            output = compute_pk(idx_sam, df_params=df_params, ndens_target=ndens_target,
                                tag_pk=tag_pk,
                                args_power=args_power, overwrite=overwrite)
            outputs.append(output)
    end = time.time()
    outputs = np.array(outputs)
    n_success = np.sum(outputs==0)
    print(f"Took {(end-start)/60} min to compute {n_success} pks with N={n_threads} threads")
        


def compute_pk(idx_sam, df_params=None, ndens_target=None, tag_pk=None,
               args_power=None, overwrite=False):

    assert ndens_target is not None or args_power is not None or df_params is not None, "Must pass df_params and ndens_target and args_power!"

    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    redshift = 0
    
    fn_dat = f"{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5"
    if not os.path.isfile(fn_dat):
        print(f"[SAM LH {idx_sam}] File {fn_dat} does not exist! Moving on")
        return 1

    fn_pk = f'../data/pks/pks{tag_pk}/pk_LH_{idx_sam}.npy'
    if os.path.isfile(fn_pk) and not overwrite:
        print(f"[SAM LH {idx_sam}] Pk {fn_pk} already exists and overwrite={overwrite}! Moving on")
        return 1

    Omega_m = df_params.loc[idx_sam, 'Omega_m']
    sigma_8 = df_params.loc[idx_sam, 'sigma_8']
    cosmo = setup_cosmology(Omega_m, sigma_8)
    args_power['cosmology'] = cosmo

    vol_hMpc = 100**3 # units Mpc/h!! 
    n_target = int(ndens_target * vol_hMpc)
    with h5py.File(fn_dat, 'r') as f:
        mstar_raw = np.array(f['mstar'])
        i_target = np.argsort(mstar_raw)[::-1][:n_target] # order by mstar and take largest to smallest to get desired ndens
        x_arr, y_arr, z_arr = f['x_position'], f['y_position'], f['z_position']
        pos_arr = np.array([x_arr, y_arr, z_arr]).T
        pos_arr = pos_arr[i_target]
        # now in Mpc, put in h^-1 Mpc
        # X*0.7 Mpc/h = X Mpc * 0.7/h
        pos_arr *= cosmo.pars['hubble']
        pk = bacco.statistics.compute_powerspectrum(pos=pos_arr, **args_power)
        np.save(fn_pk, pk)

    return 0


def setup_cosmology(Omega_m, sigma_8):
    a_factor = 1
    Ob = 0.049
    hubble = 0.6711
    ns = 0.9624
    cosmopars = dict(
            omega_cdm=Omega_m-Ob,
            omega_baryon=Ob, 
            hubble=hubble, 
            ns=ns, 
            sigma8=sigma_8,
            #tau=0.0561,
            A_s=None,
            neutrino_mass=0.,
            w0=-1,
            wa=0,
            tag="cosmo_CAMELS-SAM"
        )
        
    # access parameter dict with cosmo.pars
    cosmo = bacco.Cosmology(**cosmopars)
    cosmo.set_expfactor(a_factor)
    return cosmo


def setup_bacco():
    ##### DEFINE QUIJOTE COSMOLOGY ############
    # The cosmology is not really needed but
    # bacco P(k) corrects some modes according
    # to their cosmology, so better have it
    # just in case

    hubble = 0.6711
    ngrid = 256 #1024 #512 #256 #128 #256 #1400
    #BoxSize=100/hubble
    BoxSize = 100 #Mpc/h
    args_power = {'ngrid':ngrid,
            'box':BoxSize,
            #'cosmology':cosmo_Quijote,
            'interlacing':True,
            'kmin':0.01,
            'kmax':1.0,
            'nbins':32,
            'correct_grid':True,
            'log_binning':True,
            'deposit_method':'cic',
            'compute_correlation':False,
            'zspace':False,
            'normalise_grid': True,
            'compute_power2d':False}

    bacco.configuration.update({'pknbody' : {'ngrid'  :  ngrid}})
    # bacco.configuration.update({'pknbody' : {'log_binning' : True}})
    # bacco.configuration.update({'pknbody' : {'log_binning_kmax' : 0.99506136}})#
    # bacco.configuration.update({'pknbody' : {'log_binning_nbins' : 100}})
    # bacco.configuration.update({'pknbody' : {'min_k' : 0.01721049}})
    # bacco.configuration.update({'pk' : {'maxk' : 0.99506136}}) 
    bacco.configuration.update({'pknbody' : {'interlacing' : True}})

    bacco.configuration.update({'pknbody' : {'depmethod' : 'cic'}})

    bacco.configuration.update({'nonlinear' : {'concentration' : 'ludlow16'}})

    bacco.configuration.update({'number_of_threads' : 12})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : ngrid}})

    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})

    logger = logging.getLogger("bacco.power")
    # only log really bad events
    logger.setLevel(logging.ERROR)

    return args_power


if __name__=='__main__':
    main()
