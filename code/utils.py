import numpy as np 
import os

import baccoemu


def setup_cosmo_emu(cosmo='quijote'):
    print("Setting up emulator cosmology")
    if cosmo=='quijote':
        cosmo_params = {
            'omega_cold'    :  0.3175,
            'sigma8_cold'   :  0.834,
            'omega_baryon'  :  0.049,
            'ns'            :  0.9624,
            'hubble'        :  0.6711,
            'neutrino_mass' :  0.0,
            'w0'            : -1.0,
            'wa'            :  0.0,
            'expfactor'     :  1.0
        }

    elif cosmo=='CAMELS-SAM':
        Ob = 0.049
        hubble = 0.6711
        ns = 0.9624
        cosmo_params = {
            'omega_baryon'  :  Ob,
            'ns'            :  ns,
            'hubble'        :  hubble,
            'neutrino_mass' :  0.0,
            'w0'            : -1.0,
            'wa'            :  0.0,
            'expfactor'     :  1
        }

    else:
        raise ValueError(f'Cosmo {cosmo} not recognized!')
    return cosmo_params



def load_emu(emu_name='lbias_2.0'):
    dir_emus_lbias = '/cosmos_storage/cosmosims/data_share'
    dir_emus_mpk = '/cosmos_storage/cosmosims/datashare'
    if emu_name=='lbias_public':
        emu = baccoemu.Lbias_expansion(verbose=False)
    elif emu_name=='lbias_2.0':
        fn_emu = f'{dir_emus_lbias}/lbias_emulator/lbias_emulator2.0.0'
        emu = baccoemu.Lbias_expansion(verbose=False, 
                                    nonlinear_emu_path=fn_emu,
                                    nonlinear_emu_details='details.pickle',
                                    nonlinear_emu_field_name='NN_n',
                                    nonlinear_emu_read_rotation=False)
    elif emu_name=='mpk':
        standardspace_folder = f'{dir_emus_mpk}/mpk_baccoemu_new/mpk_oldsims_standard_emu_npca7_neurons_400_400_dropout_0.0_bn_False/'
        emu = baccoemu.Matter_powerspectrum(nonlinear_emu_path=standardspace_folder, 
                                                     nonlinear_emu_details='details.pickle')
    elif emu_name=='mpk_extended':
        extendedspace_folder = f'{dir_emus_mpk}/mpk_baccoemu_new/mpk_extended_emu_npca_20_batch_size_256_nodes_400_400_dropout_0.0_batch_norm_False/'
        emu = baccoemu.Matter_powerspectrum(nonlinear_emu_path=extendedspace_folder, 
                                            nonlinear_emu_details='details.pickle')
        
    else:
        raise ValueError(f'Emulator {emu_name} not recognized!')
    emu_param_names = emu.emulator['nonlinear']['keys']
    emu_bounds =  emu.emulator['nonlinear']['bounds']
    return emu, emu_bounds, emu_param_names
