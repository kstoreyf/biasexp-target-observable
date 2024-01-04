from functools import partial
import h5py
import logging
import multiprocessing as mp
import numpy as np
import os
import time

import bacco


def main():

    args_power = setup_bacco()
    
    #idxs_sam = np.arange(0, 999)
    #idxs_sam = np.arange(50, 1000)
    redshift = 0
    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    idxs_sam = [idx_sam for idx_sam in np.arange(0, 1000) \
                if os.path.isfile(f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5')]
    compute_pks_loop(idxs_sam, args_power, overwrite=False, n_threads=12)


def compute_pks_loop(idxs_sam, args_power, overwrite=False, n_threads=2):
    
    start = time.time()
    if n_threads>1:
        pool = mp.Pool(processes=n_threads)
        print("Starting multiprocessing pool")
        outputs = pool.map(partial(compute_pk, args_power=args_power, overwrite=overwrite), idxs_sam)
        print("Done!")
    else:
        outputs = []
        for idx_sam in idxs_sam:
            output = compute_pk(idx_sam, args_power=args_power, overwrite=overwrite)
            outputs.append(output)
    end = time.time()
    n_success = np.sum(outputs==0)
    print(f"Took {(end-start)/60} min to compute {n_success} pks with N={n_threads} threads")
        


def compute_pk(idx_sam, args_power=None, overwrite=False):

    assert args_power is not None, "Must pass args_power!"

    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    redshift = 0
    
    fn_dat = f"{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5"
    if not os.path.isfile(fn_dat):
        print(f"[SAM LH {idx_sam}] File {fn_dat} does not exist! Moving on")
        return 1

    fn_pk = f'../data/pks/pk_LH_{idx_sam}.npy'
    if os.path.isfile(fn_pk) and not overwrite:
        print(f"[SAM LH {idx_sam}] Pk {fn_pk} already exists and overwrite={overwrite}! Moving on")
        return 1

    with h5py.File(fn_dat, 'r') as f:
        x_arr, y_arr, z_arr = f['x_position'], f['y_position'], f['z_position']
        pos_arr = np.array([x_arr, y_arr, z_arr]).T
        pk = bacco.statistics.compute_powerspectrum(pos=pos_arr, **args_power)
        np.save(fn_pk, pk)

    return 0


def setup_bacco():
    ##### DEFINE QUIJOTE COSMOLOGY ############
    # The cosmology is not really needed but
    # bacco P(k) corrects some modes according
    # to their cosmology, so better have it
    # just in case

    a_Quijote = 1
    Ob = 0.049
    Om = 0.3175
    hubble = 0.6711
    ns = 0.9624
    sigma8 = 0.834
    cosmopars = dict(
            omega_cdm=Om-Ob,
            omega_baryon=Ob, 
            hubble=hubble, 
            ns=ns, 
            sigma8=sigma8,
            #tau=0.0561,
            A_s=None,
            neutrino_mass=0.,
            w0=-1,
            wa=0,
            tag="cosmo_BOSS"
        )
        
    cosmo_Quijote = bacco.Cosmology(**cosmopars)
    cosmo_Quijote.set_expfactor(a_Quijote)

    ngrid = 256 #1024 #512 #256 #128 #256 #1400
    BoxSize=100/hubble
    args_power = {'ngrid':ngrid,
            'box':BoxSize,
            'cosmology':cosmo_Quijote,
            'interlacing':True,
            'kmin':0.1,
            'kmax':1.0,
            'nbins':16,
            'correct_grid':True,
            'log_binning':False,
            'deposit_method':'cic',
            'compute_correlation':False,
            'zspace':False,
            'normalise_grid': True,
            'compute_power2d':False}

    bacco.configuration.update({'pknbody' : {'ngrid'  :  ngrid}})
    bacco.configuration.update({'pknbody' : {'log_binning' : True}})
    bacco.configuration.update({'pknbody' : {'log_binning_kmax' : 0.99506136}})#
    bacco.configuration.update({'pknbody' : {'log_binning_nbins' : 100}})
    bacco.configuration.update({'pknbody' : {'min_k' : 0.01721049}})
    bacco.configuration.update({'pk' : {'maxk' : 0.99506136}}) 
    bacco.configuration.update({'pknbody' : {'interlacing' : True}})

    bacco.configuration.update({'pknbody' : {'depmethod' : 'cic'}})

    bacco.configuration.update({'nonlinear' : {'concentration' : 'ludlow16'}})

    bacco.configuration.update({'number_of_threads' : 1})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : ngrid}})

    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})

    logger = logging.getLogger("bacco.power")
    # only log really bad events
    logger.setLevel(logging.ERROR)

    return args_power


if __name__=='__main__':
    main()
