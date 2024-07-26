import bacco

cosmo = bacco.Cosmology(verbose=False, **bacco.cosmo_parameters.Millennium)
print(cosmo.pars)

n_grid = 128 #??
box_size = 500.
LPT_order = 2 #?? should be 1 or 2?

sim = bacco.utils.create_lpt_simulation(cosmo, box_size, Nmesh=n_grid, 
                                        Seed=100672,
                                        #Seed=seed_lpt,
                                        FixedInitialAmplitude=False, InitialPhase=0, 
                                        expfactor=1.0, LPT_order=LPT_order, order_by_order=None,
                                        phase_type=1, 
                                        #ngenic_phases=False, # default
                                        ngenic_phases=True,
                                        return_disp=False, 
                                        sphere_mode=0)

fields_sim = sim.get_linear_field(ngrid=n_grid)
print(fields_sim.shape)

overdensity = fields_sim[0]

print(overdensity[0,0,:10])