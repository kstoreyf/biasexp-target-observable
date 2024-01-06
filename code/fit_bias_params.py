from functools import partial
import multiprocessing as mp
import numpy as np
import os
import scipy
import time

import baccoemu


def main():
    
    #idxs_sam = np.arange(6, 8)
    redshift = 0
    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    idxs_sam = [idx_sam for idx_sam in np.arange(0, 1000) \
                if os.path.isfile(f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5')]
    fit_bias_params_loop(idxs_sam, n_threads=12, overwrite=False)


# Followed this to get multiprocessing with emulator to work
# (using normal global not in this initializer function did not work)
# https://stackoverflow.com/questions/18778187/multiprocessing-pool-with-a-global-variable
def initializer():
    print("Loading emulator")
    global emulator
    emulator = baccoemu.Lbias_expansion()


def fit_bias_params_loop(idxs_sam, n_threads=2, overwrite=False):
    
    cosmo_params = setup_cosmo_emu()

    start = time.time()
    if n_threads>1:
        pool = mp.Pool(processes=n_threads, initializer=initializer)
        print("Starting multiprocessing pool")
        outputs = pool.map(partial(fit_bias_params,
                             cosmo_params=cosmo_params, overwrite=overwrite), idxs_sam)
        print("Done!")
    else:
        print("Starting serial loop")
        outputs = []
        for idx_sam in idxs_sam:
            output = fit_bias_params(idx_sam, emulator=emulator, cosmo_params=cosmo_params, overwrite=overwrite)
            outputs.append(output)
    end = time.time()
    n_success = np.sum(outputs==0)
    print(f"Took {(end-start)/60} min to fit {n_success} bias param sets with N={n_threads} threads")


def fit_bias_params(idx_sam, cosmo_params=None, overwrite=False):

    assert cosmo_params is not None, "Must pass emulator and cosmo_params!"

    fn_pk = f'../data/pks/pk_LH_{idx_sam}.npy'
    if not os.path.isfile(fn_pk):
        print(f"[SAM LH {idx_sam}] P(k) file {fn_pk} does not exist! Moving on")
        return 1

    fn_bp = f'../data/bias_params_bestfit/bias_params_LH_{idx_sam}.npy'
    if os.path.isfile(fn_bp) and not overwrite:
        print(f"[SAM LH {idx_sam}] Bias param file {fn_bp} already exists and overwrite={overwrite}! Moving on")
        return 1

    print("Loading pk, setting up cov, etc")
    pk = np.load(fn_pk, allow_pickle=True).item()
    
    k_sam_all = pk['k']
    i_bins = k_sam_all < 0.75 #bc emulator can't go above this
    k_sam = k_sam_all[i_bins]
    C_inv = np.diag(np.ones(len(k_sam))/len(k_sam))
    pk_sam = pk['pk'][i_bins]

    print(f"Fitting SAM {idx_sam}")
    bias_param_names = ['b1', 'b2', 'bs2', 'bl']
    bounds = get_bounds(bias_param_names)
    bias_params_0 = [0.5, 0.5, 1.0, -1.0]
    res = scipy.optimize.minimize(ln_like, bias_params_0, bounds=bounds, 
                                  args=(k_sam, pk_sam, C_inv, emulator, cosmo_params))
    
    if res['success']:
        print(f"Fit for SAM {idx_sam} terminated successfully!")
        bias_params_fit_dict = dict(zip(bias_param_names, res['x']))
        np.save(fn_bp, bias_params_fit_dict)
        return 0
    else:
        print(f"WARNING: Oh no, optimizer failed for SAM {idx_sam}! not saving params")
        return 1


def ln_like(bias_params, k_data, pk_data, C_inv, 
            emulator, cosmo_params):
    _, p_gg, _ = emulator.get_galaxy_real_pk(bias=bias_params, k=k_data, 
                                             **cosmo_params)
    delta_y = pk_data - p_gg
    lnlk = 0.5 * delta_y.T @ C_inv @ delta_y
    return lnlk


def setup_cosmo_emu():
    print("Setting up emulator cosmology")
    Ob = 0.049
    Om = 0.3175
    hubble = 0.6711
    ns = 0.9624
    sigma8 = 0.834
    cosmo_params = {
        'omega_cold'    :  Om,
        'sigma8_cold'   :  sigma8, # if A_s is not specified
        'omega_baryon'  :  Ob,
        'ns'            :  ns,
        'hubble'        :  hubble,
        'neutrino_mass' :  0.0,
        'w0'            : -1.0,
        'wa'            :  0.0,
        'expfactor'     :  1
    }
    return cosmo_params


def get_bounds(bias_param_names):
    bias_bounds = {'b1': [-1, 2],
                'b2': [-1, 2],
                'bs2': [-3.5, 3.5],
                'bl': [-5, 14],
                } 
    bounds = [bias_bounds[bname] for bname in bias_param_names]
    return bounds


if __name__=='__main__':
    main()
