import numpy as np


dir_base_input = '/dipc/kstoreyf/external/LGalaxies2020_PublicRepository/input'
minimal = True
options_minimal = ['H2FractionRecipe', 'SFRtdyn', 'Clumpingfactor', 'GasInflowVel',
                   'Dust_tExch', 'Dust_tAcc0', 'Cmax_CO', 
                   'FracDustSNIItoHot', 'FracDustSNIatoHot', 'FracDustAGBtoHot']

#vals_snII = np.arange(0.1, 1.0, 0.1)
#print(vals_snII)
#nueject_fiducial = 5.5
#vals_nueject = nueject_fiducial * np.logspace(np.log10(0.25), np.log10(4), 9)
#vals_nueject = [0.5]


#print(np.array([f'{val:.2f}' for val in vals_nueject]))
#for val_snII in vals_snII:

# via input/MCMC_inputs/MCMCParameterPriorsAndSwitches.txt
# min: 1.e-8  max: 1.e-1
kagn_fiducial = 0.0025
#vals_kagn = kagn_fiducial * np.logspace(np.log10(0.25), np.log10(4), 9)
# vals_kagn = np.logspace(np.log10(1e-8), np.log10(1e-1), 8) #8 to get nice values
# vals_kagn = np.concatenate((vals_kagn, np.array([kagn_fiducial]))) #add in fiducial
vals_kagn = np.array([kagn_fiducial])
print(np.array([f'{val:.2e}' for val in vals_kagn]))


#for val_nueject in vals_nueject:
for val_kagn in vals_kagn:
   
    #tag_lgal = f'_FracZSNIItoHot{val_snII:.1f}'
    #tag_lgal = f'_FeedbackEjectionEfficiency{val_nueject:.2f}'
    tag_lgal = f'_AgnEfficiency{val_kagn:.2e}'
    if minimal:
        tag_lgal += '_minimal'
    
    fn_orig = f'{dir_base_input}/input_MR_W1_PLANCK_LGals2020_MM.par'
    fn_new = f'{dir_base_input}/input_MR_W1_PLANCK_LGals2020_MM{tag_lgal}.par'
    print(fn_new)
    
    #find-and-replace code: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
    # Read in orig file
    with open(fn_orig, 'r') as file:
        filedata = file.read()

    # careful, this whitespace needs to be tabs! 
    # snII_orig = 'FracZSNIItoHot				0.9'
    # snII_new = f'FracZSNIItoHot				{val_snII:.1f}'
    # filedata = filedata.replace(snII_orig, snII_new)
    
    # # FeedbackEjectionEfficiency
    # nueject_orig = 'FeedbackEjectionEfficiency  5.5'
    # nueject_new = f'FeedbackEjectionEfficiency  {val_nueject:.2f}'
    # filedata = filedata.replace(nueject_orig, nueject_new)

    # AgnEfficiency
    kagn_orig = 'AgnEfficiency               0.0025'
    kagn_new = f'AgnEfficiency               {val_kagn:.2e}'
    filedata = filedata.replace(kagn_orig, kagn_new)
    
    out_orig = 'OutputDir		 ./output/'
    out_new = f'OutputDir		 ./output/output{tag_lgal}/'
    filedata = filedata.replace(out_orig, out_new)
    
    if minimal:
        for option in options_minimal:
            option_orig = f'{option}'
            option_new = f'%%%{option}'
            filedata = filedata.replace(option_orig, option_new)
    
    #Write the file out again
    with open(fn_new, 'w') as file:
        file.write(filedata)
    