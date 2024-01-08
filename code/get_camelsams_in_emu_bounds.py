import numpy as np
import os
import pandas as pd

import baccoemu


def main():
    redshift = 0
    dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'
    idxs_sam = [idx_sam for idx_sam in np.arange(0, 1000) \
                if os.path.isfile(f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5')]

    emu = load_emulator()
    fn_params = '../data/params_CAMELS-SAM.dat'
    df_params = pd.read_csv(fn_params, index_col='idx_LH')

    idxs_sam_inbounds = check_in_emu_bounds(idxs_sam, df_params, emu)
    
    fn_idxs = '../data/idxs_camelssam_in_emu_bounds.dat'
    np.savetxt(fn_idxs, idxs_sam_inbounds, fmt='%i')


def load_emulator():
    print("Loading emulator")
    emu = baccoemu.Lbias_expansion()
    return emu


def check_in_emu_bounds(idxs_sam, df_params, emu):
    Omega_m = df_params.loc[idxs_sam, 'Omega_m']
    sigma_8 = df_params.loc[idxs_sam, 'sigma_8']
    param_keys = emu.emulator['nonlinear']['keys']
    i_Omega_m = param_keys.index('omega_cold')
    i_sigma_8 = param_keys.index('sigma8_cold')
    emu_bounds = emu.emulator['nonlinear']['bounds']
    i_inbounds = (Omega_m >= emu_bounds[i_Omega_m][0]) & \
                 (Omega_m <= emu_bounds[i_Omega_m][1]) & \
                 (sigma_8 >= emu_bounds[i_sigma_8][0]) & \
                 (sigma_8 <= emu_bounds[i_sigma_8][1])
    return np.array(idxs_sam)[i_inbounds]


if __name__=='__main__':
    main()
