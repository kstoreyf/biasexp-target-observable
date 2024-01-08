mport h5py
import numpy as np
import os
import pandas as pd
import subprocess
import time

#to tar, in notebook: !tar -czvf CAMELS-SAM_data_rest.tar.gz CAMELS-SAM_data

def main():
    #idxs_sam = np.arange(0,1000)
    idxs_sam = np.arange(0,10)

    dir_sams = '/home/jovyan/Data/SCSAM/LH'

    params_sam = []
    for idx_sam in idxs_sam:
        
        fn_params = f'{dir_sams}/LH_{idx_sam}/CosmoAstro_params.txt'
        params = np.loadtxt(fn_params)
        params = params[:-1] # last param is vestigial, doesn't vary
        #params = np.concatenate(([idx_sam], params)) 
        params_sam.append(params)
        
    fn_params_all = 'params_CAMELS-SAM.dat'
    column_names = ['Omega_m','sigma_8', 'A_SN1_x1.7', 'A_SN2_p3', 'A_AGN_x0.002']
    df = pd.DataFrame(np.array(params_sam), columns=column_names)
    #df.loc[:, 'idx_LH'] = df['idx_LH'].apply(int) #convert to an int
    print(df.shape)
    print(len(idxs_sam))
    df['idx_LH'] = idxs_sam
 
    print(df)
    df.to_csv(fn_params_all, index=False)

    
if __name__=='__main__':
    main()


