import multiprocessing as mp
import numpy as np
import scipy

import baccoemu


def main():
    
    idxs_sam = np.arange(0, 999)
    fit_bias_params_loop(idxs_sam)


def fit_bias_params_loop(idxs_sam, n_threads=2):
    
    emulator = baccoemu.Lbias_expansion()
    cosmo_params = setup_cosmo_emu()

    if n_threads>1:
        pool = mp.Pool(processes=n_threads)
        print("Starting multiprocessing pool")
        # TODO check how to pass additional args
        # TODO check whether i can pass more idxs_sam than threads 
        # and if itll be smart about holding the queue
        _ = pool.map(fit_bias_params, idxs_sam, 
                     args=(emulator, cosmo_params))
        print("Done!")
    else:
        for idx_sam in idxs_sam:
            fit_bias_params(idx_sam, emulator, cosmo_params)


def fit_bias_params(idx_sam, emulator, cosmo_params):

    fn_pk = '../data/pks/pk_LH_{idx_sam}.npy'
    fn_bp = '../data/bias_params_bestfit/bias_params_LH_{idx_sam}.npy'
    pk = np.load(fn_pk, allow_pickle=True)
    
    k_sam_all = pk['k']
    i_bins = k_sam_all < 0.75 #bc emulator can't go above this
    k_sam = k_sam_all[i_bins]
    C_inv = np.diag(np.ones(len(k_sam))/len(k_sam))
    pk_sam = pk['pk'][i_bins]

    print(f"Fitting SAM {idx_sam}")
    bounds = get_bounds()
    bias_params_0 = [0.5, 0.5, 1.0, -1.0]
    res = scipy.optimize.minimize(ln_like, bias_params_0, bounds=bounds, 
                                  args=(k_sam, pk_sam, C_inv, emulator, cosmo_params))
    if res['success']:
        np.save(fn_bp, res['x'])
    else:
        print(f"Oh no, optimizer failed for SAM {idx}! not saving params")


def ln_like(bias_params, k_data, pk_data, C_inv, 
            emulator, cosmo_params):
        _, p_gg, _ = emulator.get_galaxy_real_pk(bias=bias_params, k=k_data, 
                                                 **cosmo_params)
        delta_y = pk_data - p_gg
        lnlk = 0.5 * delta_y.T @ C_inv @ delta_y
        return lnlk


def setup_cosmo_emu():
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


def get_bounds():
    bias_param_names = ['b1', 'b2', 'bs2', 'bl']
    bias_bounds = {'b1': [-0.25, 1.75],
                'b2': [-1, 2],
                'bs2': [-3, 1],
                'bl': [-5, 2],
                } 
    bounds = [bias_bounds[bname] for bname in bias_param_names]
    return bounds


if __name__=='__main__':
    main()