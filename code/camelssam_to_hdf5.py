import h5py
import numpy as np
import os
import pandas as pd
import subprocess
import time

#to tar, in notebook: !tar -czvf CAMELS-SAM_data_n107.tar.gz CAMELS-SAM_data

def main():
    idxs_sam = np.arange(0,1000)
    #idxs_sam = [0, 1]
    #idxs_sam = [613, 257, 61, 579, 96]
    
    galprop_fields = ['redshift', 'halo_index', 'sat_type', 'mstar', 'mhalo', 'rhalo', 'sfr', 'sfrave1gyr',
                      'x_position', 'y_position', 'z_position', 'vx', 'vy', 'vz']
    redshift = 0

    dir_sams = '/home/jovyan/Data/SCSAM/LH'
    dir_dat = '/home/jovyan/CAMELS-SAM_data'
    #dir_sams = '/dipc/kstoreyf/CAMELS-SAM'
    #dir_dat = '/lscratch/kstoreyf/CAMELS-SAM_data'

    os.chdir(f'{dir_sams}/LH_{idxs_sam[0]}') 
    totaldirs=(subprocess.check_output('''ls -l sc-sam/ | grep -c ^d''', shell=True,text=True))
    totaldirs=np.float64(totaldirs)
    nsubvol=np.int64(totaldirs**(1./3.))

    dats_sam = []
    for idx_sam in idxs_sam:
        fn_dat = f'{dir_dat}/LH_{idx_sam}_galprops_z{redshift}.hdf5'
        
        if os.path.exists(fn_dat):
            print(f'File {fn_dat} already exists, skipping!')
            continue
        
        print(f'reading sam LH_{idx_sam}')
        s = time.time()
        try:
            dat_sam = ProcessSAMdat_single_redshift_lite(f'{dir_sams}/LH_{idx_sam}/sc-sam', 
                                                           nsubvol, redshift, galprop_fields, 'gal')
            print('CAMELS-SAM simulation LH',idx_sam,' at redshift ', redshift, ' has this many galaxies: ', dat_sam.shape)
            e = time.time()
            print(f'Reading time: {e-s} s')

        except FileNotFoundError as err:
            print('Missing file!! in LH', idx_sam)
            print(err)
            continue
        
        s = time.time()
        with h5py.File(fn_dat, 'w') as f:
            print('Saving data to', fn_dat)
            for i in range(len(galprop_fields)):
                if galprop_fields[i]=='redshift':
                    continue
                f.create_dataset(galprop_fields[i], data=dat_sam[:,i], chunks=True)
            print('Saved!')
        e = time.time()
        print(f'Writing time: {e-s} s')



def ProcessSAMdat_single_redshift_lite(path_to_SAM, Nsubvols, sought_z, fieldswanted, gal_or_halo):
    #NOTE: these ?_colnames are the columns, in order, of the data in the original .dat files. You can give fieldswanted in nearly any order, but the order that the columns will take in the final array will depend on the fields' order in these lists. NOTE: if you care about halo_index, birthhaloID, or roothaloID, redshift will NOT be in the 0th index; update the line below that assumes that!
    g_colnames = ['halo_index', 'birthhaloid', 'roothaloid', 'redshift', 'sat_type',
                  'mhalo', 'm_strip', 'rhalo', 'mstar', 'mbulge', 'mstar_merge', 'v_disk',
                  'sigma_bulge', 'r_disk', 'r_bulge', 'mcold', 'mHI', 'mH2', 'mHII', 'Metal_star',
                  "Metal_cold", 'sfr', 'sfrave20myr', 'sfrave100myr', 'sfrave1gyr',
                  'mass_outflow_rate', 'metal_outflow_rate', 'mBH', 'maccdot', 'maccdot_radio',
                  'tmerge', 'tmajmerge', 'mu_merge', 't_sat', 'r_fric', 'x_position',
                  'y_position', 'z_position', 'vx', 'vy', 'vz']
    h_colnames = ['halo_index', 'halo_id', 'roothaloid', 'orig_halo_ID', 'redshift', 'm_vir', 'c_nfw',
                  'spin', 'm_hot', 'mstar_diffuse', 'mass_ejected', 'mcooldot',
                  'maccdot_pristine', 'maccdot_reaccrete', 'maccdot_metal_reaccrete',
                  'maccdot_metal', 'mdot_eject', 'mdot_metal_eject', 'maccdot_radio',
                  'Metal_hot', 'Metal_ejected', 'snap_num']
    g_header_rows = []
    for i in range(0, len(g_colnames)):
        g_header_rows.append(i)
    h_header_rows = []
    for i in range(0, len(h_colnames)):
        h_header_rows.append(i)

    input_path=path_to_SAM

    if type(fieldswanted) == list:
        #print(type(fieldswanted))
        pass
    else:
        return "Fieldswanted should be a list with the fields you want as strings!"

    # fix so guaranteed to get reshift correct index
    fieldswanted_ordered = []
    if gal_or_halo=='gal':
        colnames = g_colnames
    elif gal_or_halo=='halo':
        colnames = h_colnames
    else:
        raise ValueError('gal_or_halo must be gal or halo!')
    for col in colnames:
        if col in fieldswanted:
            fieldswanted_ordered.append(col) 
    i_redshift = fieldswanted_ordered.index('redshift')
    #print(fieldswanted_ordered)
    #print(i_redshift)
    
    All_halos=np.zeros((1,len(fieldswanted)))
    if gal_or_halo=="gal":
        checknums=0
        for x_i in np.arange(0,Nsubvols,1):
            for x_j in np.arange(0,Nsubvols,1):
                for x_k in np.arange(0,Nsubvols,1):
                    #galprop = pd.read_csv('{}/{}_{}_{}/galprop_0-99.dat'.format(input_path, x_i, x_j, x_k),
                    #                      delimiter=' ', skiprows=g_header_rows, names=g_colnames)
                    # print('galprop read for ',x_i, x_j, x_k,' shape:', galprop.shape)
                    #current_galprops=galprop[fieldswanted[:]].to_numpy()
                    
                    galprop = pd.read_csv('{}/{}_{}_{}/galprop_0-99.dat'.format(input_path, x_i, x_j, x_k),
                                          delimiter=' ', skiprows=len(g_colnames), names=g_colnames, 
                                          usecols=fieldswanted)            
                    #print(galprop)
                    current_galprops=galprop.to_numpy()
                    #print(current_galprops.shape)
                    
                    #print('For subvolume ',x_i,x_j,x_k, current_galprops.shape)
                    unique_redshifts=set(current_galprops[:,i_redshift]) #update this if redshift will NOT be in the 0th column; see note above
                    unique_redshifts = np.array(sorted(unique_redshifts))
                    # print(unique_redshifts)
                    idx = (np.abs(unique_redshifts - sought_z)).argmin()
        
                    current_galprops_z=current_galprops[np.where(current_galprops[:,i_redshift][:]==unique_redshifts[idx])[0],:]
                    #print(current_galprops_z.shape)
                    checknums=checknums+len(current_galprops_z)
                    All_halos=np.concatenate((All_halos,current_galprops_z))
                    #print(All_halos.shape)
        #print(All_halos.shape, checknums)
        return All_halos
    elif gal_or_halo=="halo":
        checknums2=0
        for x_i in np.arange(0,Nsubvols,1):
            for x_j in np.arange(0,Nsubvols,1):
                for x_k in np.arange(0,Nsubvols,1):
                    haloprop = pd.read_csv('{}/{}_{}_{}/haloprop_0-99.dat'.format(input_path, x_i, x_j, x_k),
                                           delimiter=' ', skiprows=h_header_rows, names=h_colnames)
                    current_haloprops=haloprop[fieldswanted[:]].to_numpy()
                    print('For subvolume ',x_i,x_j,x_k, current_haloprops.shape)
                    unique_redshifts=set(current_haloprops[:,i_redshift])
                    unique_redshifts = np.array(sorted(unique_redshifts))
                    idx = (np.abs(unique_redshifts - sought_z)).argmin()
                    current_haloprops_z=current_haloprops[np.where(current_haloprops[:,i_redshift][:]==unique_redshifts[idx])[0],:]
                    # print(current_haloprops_z.shape)
                    checknums2=checknums2+len(current_haloprops_z)
                    All_halos=np.concatenate((All_halos,current_haloprops_z))
        print(All_halos.shape, checknums2)
        return All_halos[1:,:]
    else:
        print("gal_or_halo need to be a string, either 'gal' or 'halo', to get galprop or haloprop respectively. Make sure the fields you want are actually reflected!")
        print("Column names of galprop file: ", g_colnames)
        print("Column names of haloprop file: ", h_colnames)
        return All_halos



if __name__=='__main__':
    main()

