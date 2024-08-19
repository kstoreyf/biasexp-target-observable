import numpy as np 
import os

import bacco
import baccoemu


param_label_dict = {'omega_cold': r'$\Omega_\mathrm{cold}$',
                'sigma8_cold': r'$\sigma_{8}$',
                'sigma_8': r'$\sigma_{8}$',
                'hubble': r'$h$',
                'h': r'$h$',
                'ns': r'$n_\mathrm{s}$',
                'n_s': r'$n_\mathrm{s}$',
                'omega_baryon': r'$\Omega_\mathrm{b}$',
                'omega_m': r'$\Omega_\mathrm{m}$',
                'b1': r'$b_1$', 
                'b2': r'$b_2$', 
                'bs2': r'$b_{s^2}$', 
                'bl': r'$b_{\nabla^2 \delta}$',
                }


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


def setup_cosmo(cosmo_name, return_box_size=False):
    if cosmo_name=='millennium_planck':
        box_size = 480.279
        omega_matter = 0.315
        baryon_fraction = 0.155
        omega_baryon = omega_matter * baryon_fraction
        omega_cdm = omega_matter - omega_baryon
        mill_planck_dict = {'omega_baryon': omega_baryon, #omega_m * baryon_fraction
                    'omega_de': 0.685,
                    'hubble': 0.673,
                    'omega_cdm': omega_cdm,
                    }
        cosmo = bacco.Cosmology(verbose=False, **mill_planck_dict)
    else:
        raise ValueError(f'Cosmo {cosmo_name} not recognized!')
    
    if return_box_size:
        return cosmo, box_size
    else:
        return cosmo


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


### BIAS

bias_to_pbias_param_name_dict = {'b1': 'J2',
                                 'b2': 'J22',
                                 'bs2': 'J2=2'}

pbias_to_bias_param_name_dict = {'J2': 'b1',
                                 'J22': 'b2', 
                                 'J2=2': 'bs2'}

def pbias_params_to_bias_params(pbias_param_dict, bias_param_names=None):
    if bias_param_names is None:
        bias_param_names = [pbias_to_bias_param_name_dict[pn] for pn in pbias_param_dict.keys()]
    def _f_J2(J2):
        return J2
    def _f_J22(J22):
        return J22
    # 1/2 bJ2=2 = bK2
    def _f_J2__2(J2__2):
        return 0.5*J2__2
    relation_dict = {'J2': _f_J2,
                     'J22': _f_J22,
                     'J2=2': _f_J2__2}
    pbias_param_names = [bias_to_pbias_param_name_dict[bpn] for bpn in bias_param_names]
    bias_params = [relation_dict[pbpn](pbias_param_dict[pbpn]) for pbpn in pbias_param_names]
    return bias_params, bias_param_names


def pbias_cov_to_bias_cov(pbias_cov_dict, bias_param_names=None):
    if bias_param_names is None:
        bias_param_names = [pbias_to_bias_param_name_dict[pn] for pn in pbias_cov_dict.keys()]
    def _f_J2_err(err_J2):
        return err_J2
    def _f_J22_err(err_J22):
        return err_J22
    # 1/2 bJ2=2 = bK2
    def _f_J2__2_err(err_J2__2):
        return 0.5*err_J2__2
    relation_dict = {'J2': _f_J2_err,
                     'J22': _f_J22_err,
                     'J2=2': _f_J2__2_err}
    pbias_param_names = [bias_to_pbias_param_name_dict[bpn] for bpn in bias_param_names]
    bias_cov = [relation_dict[pbpn](pbias_cov_dict[pbpn]) for pbpn in pbias_param_names]
    return bias_cov, bias_param_names


def compute_smf(log_mstar, vol, bin_edges=None):

    if bin_edges is None:
        bin_edges = np.linspace(8, 12.5, 40)

    bin_width = bin_edges[1] - bin_edges[0]      
    bins_avg = 0.5*(bin_edges[1:] + bin_edges[:-1])          
    #bins_avg = bin_edges[0:-1] + bin_edges/2.   
    phi, _ = np.histogram(log_mstar, bins=bin_edges)   
    smf = phi / vol / bin_width
    
    return bins_avg, smf


def get_colors(colorby_arr, cmap='viridis', log=False):
    import matplotlib   
    if log:
        locs_norm = matplotlib.colors.LogNorm(np.min(colorby_arr), np.max(colorby_arr), len(colorby_arr))
    else:
        locs_norm = matplotlib.colors.Normalize(np.min(colorby_arr), np.max(colorby_arr), len(colorby_arr))
    cmap = matplotlib.colormaps[cmap]
    colors = cmap(locs_norm(colorby_arr))
    return colors, cmap, locs_norm