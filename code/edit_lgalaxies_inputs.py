import numpy as np


dir_base_input = '/dipc/kstoreyf/external/LGalaxies2020_PublicRepository/input'

vals_snII = np.arange(0.1, 1.0, 0.1)
print(vals_snII)

for val_snII in vals_snII:
    
    tag_params = f'_FracZSNIItoHot{val_snII:.1f}'
    
    fn_orig = f'{dir_base_input}/input_MR_W1_PLANCK_LGals2020_MM.par'
    fn_new = f'{dir_base_input}/input_MR_W1_PLANCK_LGals2020_MM{tag_params}.par'
    
    #find-and-replace code: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
    # Read in orig file
    with open(fn_orig, 'r') as file:
        filedata = file.read()

    # careful, this whitespace needs to be tabs! 
    snII_orig = 'FracZSNIItoHot				0.9'
    snII_new = f'FracZSNIItoHot				{val_snII:.1f}'
    filedata = filedata.replace(snII_orig, snII_new)
    

    out_orig = 'OutputDir		 ./output/'
    out_new = f'OutputDir		 ./output/output{tag_params}/'
    filedata = filedata.replace(out_orig, out_new)
    
    #Write the file out again
    with open(fn_new, 'w') as file:
        file.write(filedata)
    