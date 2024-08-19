import numpy as np
import pandas as pd


def main():

    props = ['AgnEfficiency']
    spacings = ['log']    
    n_per_dim = 10
    tag_params = f'_{props[0]}_n{n_per_dim}'
    fn_params = f'../data/params_lgal/params_lgal{tag_params}.txt'
    
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
    
    make_grid(priors, props, spacings, fn_params, n_per_dim=n_per_dim)


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



if __name__ == "__main__":
    main()