import h5py
import logging
import numpy as np
#import os
#from os import listdir
#from os.path import isfile, join

import bacco


def main():

    args_power = setup_bacco()
    
    idxs_sam = np.arange(0, 999)
    compute_pks_loop(idxs_sam, args_power)


def compute_pks_loop(idxs_sam, args_power):
    
    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    redshift = 0
    #fns_dat = [join(dir_dat, f) for f in listdir(dir_dat) if isfile(join(dir_dat, f))]

    for idx in idxs_sam:
        fn_dat = f'{dir_dat}/LH_{idx}_galprops_z{redshift}.hdf5'
        fn_pk = '../data/pks/pk_LH_{idx}.npy'
        with h5py.File(fn_dat, 'r') as f:
            x_arr, y_arr, z_arr = f['x_position'], f['y_position'], f['z_position']
            pos_arr = np.array([x_arr, y_arr, z_arr]).T
            pk = bacco.statistics.compute_powerspectrum(pos=pos_arr, **args_power)
            # TODO check that want to be saving whole pk dict, 
            # vs just pk['pk'] and pk['k']
            np.save(fn_pk, pk)


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

    bacco.configuration.update({'number_of_threads' : 12})
    bacco.configuration.update({'scaling' : {'disp_ngrid' : ngrid}})

    bacco.configuration.update({'pk':{'boltzmann_solver': 'CLASS'}})

    logger = logging.getLogger("bacco.power")
    # only log really bad events
    logger.setLevel(logging.ERROR)

    return args_power


if __name__=='__main__':
    main()