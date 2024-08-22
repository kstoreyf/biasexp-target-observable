import numpy as np
import pandas as pd


def main():

    #props = ['AgnEfficiency']
    #props = ['FeedbackEjectionEfficiency']
    # spacings = ['log']    
    # n_per_dim = 10
    # tag_params = f'_{props[0]}_n{n_per_dim}'
    # fn_params = f'../data/params_lgal/params_lgal{tag_params}.txt'
    
    tag_params = '_minmax'
    fn_params = f'../data/params_lgal/params_lgal{tag_params}.txt'

    priors = load_priors()

    #make_grid(priors, props, spacings, fn_params, n_per_dim=n_per_dim)
    make_minmax(priors, fn_params)

def load_priors():
    dtype = [
        ('Name', 'U50'),          # String with max length 50
        ('PropValue', 'f8'),      # Float64
        ('PriorMin', 'f8'),       # Float64
        ('PriorMax', 'f8'),       # Float64
        ('Type', 'U20'),          # String with max length 20
        ('Sampling_Switch', 'i4') # Integer
    ]   
    fn_priors = '/dipc/kstoreyf/external/LGalaxies2020_PublicRepository/input/MCMC_inputs/MCMCParameterPriorsAndSwitches.txt'
    priors = np.loadtxt(fn_priors, skiprows=2, dtype=dtype)
    return priors


def make_grid(priors, props, spacings, fn_params, n_per_dim=10):
    
    assert len(props)==1, "Not implemented for higher d yet!"
    param_df = pd.DataFrame()
    
    val_arr = []
    for prop, spacing in zip(props, spacings):
        i_name = np.where(priors['Name'] == prop)[0][0]
        val_min, val_max = priors['PriorMin'][i_name], priors['PriorMax'][i_name]
        if spacing == 'log':
            vals = np.logspace(np.log10(val_min), np.log10(val_max), n_per_dim)
        else:
            vals = np.linspace(val_min, val_max, n_per_dim)
        val_arr.append(vals)

    for i, prop in enumerate(props):
        param_df[prop] = val_arr[i]

    #indices = np.arange(len(param_df))
    #param_df['index'] = indices

    print(param_df)
    # val_grid = np.meshgrid(*val_arr)
    # print(val_grid.shape)
    # for prop in props:
    #     param_df[prop] = vals
    param_df.to_csv(fn_params)
    print(f'Saved to {fn_params}')
    
    
def make_minmax(priors, fn_params):
        
    props = [priors['Name'][i] for i in range(len(priors['Name'])) if priors['Type'][i]=='Physical']


    dict_fid = {priors['Name'][i]: priors['PropValue'][i] for i in range(len(priors['Name'])) 
                                                         if priors['Type'][i]=='Physical'}
    print(dict_fid)
    dict_fid['name_iparam'] = 'fiducial'
    
    n_tot = 2*len(props) + 1 # +1 for fiducial; will be at end
    param_df = pd.DataFrame([dict_fid] * n_tot)
    i_row = 1 # makes fiducial first 
    
    for prop in props:
        i_name = np.where(priors['Name'] == prop)[0][0]
        val_min, val_max = priors['PriorMin'][i_name], priors['PriorMax'][i_name]
        param_df.loc[i_row, prop] = val_min
        param_df.loc[i_row, "name_iparam"] = f"{prop}_min"
        param_df.loc[i_row+1, prop] = val_max
        param_df.loc[i_row+1, "name_iparam"] = f"{prop}_max"
        i_row += 2
        
    print(param_df)
    param_df.to_csv(fn_params)
    print(f'Saved to {fn_params}')



if __name__ == "__main__":
    main()