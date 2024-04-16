import numpy as np


dir_base_ICs = '/dipc/kstoreyf/CAMELS-SAM_ICS/LH'
dir_2lpt = '/dipc/kstoreyf/external/2lpt'

n_start = 0
n_tot = 1000
idxs_LH = np.arange(n_start, n_tot)

for idx_LH in idxs_LH:
    fn_2lpt_orig = f'{dir_base_ICs}/LH_{idx_LH}/2LPT.param'
    fn_2lpt_new = f'{dir_base_ICs}/LH_{idx_LH}/2LPT_new.param'
    
    #find-and-replace code: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
    # Read in orig file
    with open(fn_2lpt_orig, 'r') as file:
        filedata = file.read()

    # Replace glass file location
    fn_glass_orig = '/mnt/ceph/users/lperez/ICsSAM/dummy_glass_dmonly_64.dat'
    fn_glass_new = f'{dir_2lpt}/dummy_glass_dmonly_64.dat'
    filedata = filedata.replace(fn_glass_orig, fn_glass_new)

    # Modifies both OutputDir and FileWithInputSpectrum (Pk_m_z=0.000.txt location)
    dir_output_orig = './'
    dir_output_new = f'{dir_base_ICs}/LH_{idx_LH}/'
    filedata = filedata.replace(dir_output_orig, dir_output_new)
    
    # Modifies NumFilesWrittenInParallel
    dir_numfiles_orig = 'NumFilesWrittenInParallel 20'
    dir_numfiles_new = 'NumFilesWrittenInParallel 12' # match number of cores will run it on
    filedata = filedata.replace(dir_numfiles_orig, dir_numfiles_new)
    
    # Write the file out again
    with open(fn_2lpt_new, 'w') as file:
        file.write(filedata)
    