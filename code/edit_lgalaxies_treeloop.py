import numpy as np
import pandas as pd
from pathlib import Path
import re


def main():
    dir_base_input = '/dipc/kstoreyf/external/LGalaxies2020_PublicRepository/input'

    check_mode = False

    #tag_params = '_AgnEfficiency_n10'
    #tag_lgal = '_DM_fasttesting'
    
    tag_params = ''
    tag_lgal = '_DM_orig_treeloop'

    n_treefiles_max = 512
    n_blocks = 8
    # n_treefiles_max = 16
    # n_blocks = 1
    n_trees_per_block = n_treefiles_max // n_blocks

    if tag_params is not '':
        param_df = pd.read_csv(f'../data/params_lgal/params_lgal{tag_params}.txt', index_col=0)
        print(param_df.columns)
        print(param_df)
        n_iparams = len(param_df)
    else:
        n_iparams = 1
        
    dir_input = f'{dir_base_input}/inputs{tag_lgal}{tag_params}'
    Path(dir_input).mkdir(parents=True, exist_ok=True)

    # via input/MCMC_inputs/MCMCParameterPriorsAndSwitches.txt
    # min: 1.e-8  max: 1.e-1
    # vals_kagn = np.logspace(np.log10(1e-8), np.log10(1e-1), 8) #8 to get nice values
    # print(np.array([f'{val:.2e}' for val in vals_kagn]))

    # Replace property values
    for i_param in range(n_iparams):
        
        if tag_params is not '':
            tag_iparam = f'_iparam{i_param}'
        else:
            tag_iparam = ''
        
        fn_orig = f'{dir_base_input}/input_MR_W1_PLANCK_LGals2020_DM_tocopy.par'

        #find-and-replace code: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
        # Read in orig file
        with open(fn_orig, 'r') as file:
            filedata = file.read()
                 
        # TODO update the below replacement function to work for not just numbers, use here too
        # change the output dir
        out_orig = 'OutputDir		  ./output/output_DM_test'
        out_new = f'OutputDir		  ./output/output{tag_lgal}{tag_params}'
        filedata = filedata.replace(out_orig, out_new)


        # change the filename, to go with the iparam
        fng_orig = 'FileNameGalaxies          SA_DM_test3'
        fng_new = f'FileNameGalaxies          SA_DM{tag_iparam}'
        filedata = filedata.replace(fng_orig, fng_new)
        
        # loop thru the properties that we want to change, change the file data
        if tag_params is not '':
            for prop in param_df.columns:
                val = param_df[prop][i_param]
                filedata = replace_property(filedata, prop, val)
        
        # if we want to break the tree files up, do it here  
        for first_treefile in range(0, n_treefiles_max, n_trees_per_block):
    
            last_treefile = first_treefile + n_trees_per_block - 1
            tag_trees = f'_tree{first_treefile}-{last_treefile}'

            fn_new = f'{dir_input}/input_MR_W1_PLANCK_LGals2020{tag_lgal}{tag_iparam}{tag_trees}.par'
            print(fn_new)
        
            #print(first_treefile, last_treefile)
            if check_mode:
                continue

            filedata = replace_property(filedata, 'FirstFile', first_treefile)
            filedata = replace_property(filedata, 'LastFile', last_treefile)
            
            #print(filedata)
            #Write the file out again
            with open(fn_new, 'w') as file:
                file.write(filedata)
            
    
    
def replace_property(text, property_name, val_new):
    # Construct the regex pattern to match the property name followed by any whitespace and the number
    pattern = fr'({property_name})(\s+)([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    if match:
        # Extract and return the matched parts
        name = match.group(1)
        whitespace = match.group(2)
        val_orig = match.group(3)
        
        segment_orig = name + whitespace + val_orig
        segment_new = name + whitespace + str(val_new)
        
        text = text.replace(segment_orig, segment_new)
        return text
    
    else:
        raise ValueError(f'Property {property_name} not found in text')
    
    
if __name__ == '__main__':
    main()