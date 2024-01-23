import numpy as np
import sys,os
from pyDOE import *
from mpi4py import MPI
import camb

NameForSims = str(sys.argv[1])
NumberToGen = np.int(sys.argv[2])

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

#################################### INPUT #############################################
root_out   = '/mnt/ceph/users/lperez/ICsSAM'
dimensions = 5      #number of cosmo+astro parameters
points     = NumberToGen   #number of simulations
seed       = 2      #seed for the latin-hypercube
fout_lh    = 'CosmoAstro_params_hypercube.txt'
#params_file = np.loadtxt('/mnt/ceph/users/lperez/ICsSAM/CosmoAstroSeed_params_SIMBA.txt', usecols=[1,2])

# CAMB parameters
Omega_b      = 0.049
h            = 0.6711
ns           = 0.9624
hierarchy    = 'degenerate'
Mnu          = 0.0 #eV
Nnu          = 0   #number of massive neutrinos
Neff         = 3.046
As           = 2.13e-9
tau          = None
Omega_k      = 0.0
pivot_scalar = 0.05
pivot_tensor = 0.05
kmax         = 200.0
k_per_logint = 20
redshifts    = [0]

#              [Om,   s8,   astro1, astro2, astro3, astro4]
#              [Om,   s8,   epsilon_SN0, alpha_rh, fedd]
Min = np.array([0.1,  0.6,  0.25*1.7,   3.0-2.0,   0.25*0.002])
Max = np.array([0.5,  1.0,  4.00*1.7,   3.0+2.0,   4.0*0.002])


########################################################################################

# set logscale for astro params, except for the additive one
Min[2] = np.log10(Min[2])
Max[2] = np.log10(Max[2])
Min[4:] = np.log10(Min[4:])
Max[4:] = np.log10(Max[4:])

# get latin hypercube
np.random.seed(seed)
coords = lhs(dimensions, samples=points, criterion='c')

# generate the parameters
params = np.zeros((points,dimensions), dtype=np.float64)
params = Min + coords*(Max-Min)
params[:,2] = 10**params[:,2]
params[:,4:] = 10**params[:,4:]
print 'Newly generated ones: ', params
# print '%.5f < Om < %.5  bf : <Om> = %.5f'\
#     %(np.min(params[:,0]), np.max(params[:,0]), np.mean(params[:,0]))
# print '%.5f < s8 < %.5f : <s8> = %.5f'\
#     %(np.min(params[:,1]), np.max(params[:,1]), np.mean(params[:,1]))
# print '%.5f < Astro1 < %.5f : <Astro1> = %.5f'\
#     %(np.min(params[:,2]), np.max(params[:,2]), 10**(np.mean(np.log10(params[:,2]))))
# print '%.5f < Astro2 < %.5f : <Astro2> = %.5f'\
#     %(np.min(params[:,3]), np.max(params[:,3]), 10**(np.mean(np.log10(params[:,3]))))
# print '%.5f < Astro3 < %.5f : <Astro3> = %.5f'\
#     %(np.min(params[:,4]), np.max(params[:,4]), 10**(np.mean(np.log10(params[:,4]))))
# print '%.5f < Astro4 < %.5f : <Astro4> = %.5f'\
#     %(np.min(params[:,5]), np.max(params[:,5]), 10**(np.mean(np.log10(params[:,5]))))
# print Min
# print Max
# print ' '

print ('Om: ',np.min(params[:,0]), np.max(params[:,0]), np.mean(params[:,0]))
print ('s8: ',np.min(params[:,1]), np.max(params[:,1]), np.mean(params[:,1]))
print ('epsilon_SN0: ',np.min(params[:,2]), np.max(params[:,2]), 10**(np.mean(np.log10(params[:,2]))))
print ('alpha_rh: ',np.min(params[:,3]), np.max(params[:,3]), np.mean(params[:,3]))
print ('fedd: ',np.min(params[:,4]), np.max(params[:,4]), 10**(np.mean(np.log10(params[:,4]))))
# print ('epsilon_windQSO: ',np.min(params[:,5]), np.max(params[:,5]), 10**(np.mean(np.log10(params[:,5]))))


## check if the generated parameters are the same as the saved ones
#if os.path.exists(fout_lh):
#    params_file = np.loadtxt(fout_lh, usecols=[1,2])
#    equal = (np.around(params,5)==params_file) #approximate params with 5 decimals
#    if np.any(equal==False):  raise Exception('Cosmological parameters are different!!!')
#else:
#    np.savetxt(fout_lh, params, fmt='%.5f')
#
# get the numbers each cpu will work on
numbers = np.where(np.arange(points)%nprocs==myrank)[0]

Num=0

#for i in numbers:
for i in np.arange(710,711,1):

    # create output folder in case it does not exists
    folder = '%s/cLH%s/'%(root_out,i)
    if not(os.path.exists(folder)):  os.system('mkdir %s'%folder)

    # create initial conditions folder if it does not exists
    folder_ICs = '%s/ICs'%folder
    if not(os.path.exists(folder_ICs)):  os.system('mkdir %s'%folder_ICs)

    # find the values of the cosmological parameters
    Omega_m = params[i,0]
    s8      = params[i,1]
    astro1  = params[i,2]
    astro2  = params[i,3]
    astro3  = params[i,4]
    astro4  = 0.002
    # Omega_m = Extremes[i,0]
    # s8      = Extremes[i,1]


    g = open('%s/CosmoAstro_params_recheck.txt'%folder, 'w')
    g.write('%.5f %.5f %.5f %.5f %.5f %.5f\n'\
            %(Omega_m, s8, astro1, astro2, astro3, astro4))
    # g.write('%.5f %.5f\n'\
    #         %(Omega_m, s8))
    g.close()

    print 'realization %d'%i
    print 'Omega_m = %.5f'%Omega_m
    print 's8      = %.5f'%s8
    print 'Astro1  = %.5f'%astro1
    print 'Astro2  = %.5f'%astro2
    print 'Astro3  = %.5f'%astro3
    print 'Astro4  = %.5f'%astro4

    ##### run CAMB #####
    Omega_c  = Omega_m - Omega_b
    pars     = camb.CAMBparams()

    # set accuracy of the calculation
    # pars.set_accuracy(AccuracyBoost=4.0, lSampleBoost=4.0, lAccuracyBoost=4.0,
    #                   HighAccuracyDefault=True, DoLateRadTruncation=True)
    pars.set_accuracy(AccuracyBoost=4.0, lSampleBoost=4.0, lAccuracyBoost=4.0,
                      DoLateRadTruncation=True)

    # set value of the cosmological parameters
    pars.set_cosmology(H0=h*100.0, ombh2=Omega_b*h**2, omch2=Omega_c*h**2,
                       mnu=Mnu, omk=Omega_k, neutrino_hierarchy=hierarchy,
                       num_massive_neutrinos=Nnu, nnu=Neff, tau=tau)

    # set the value of the primordial power spectrum parameters
    pars.InitPower.set_params(As=As, ns=ns,
                              pivot_scalar=pivot_scalar, pivot_tensor=pivot_tensor)

    # set redshifts, k-range and k-sampling
    pars.set_matter_power(redshifts=redshifts, kmax=kmax, k_per_logint=k_per_logint)

    # compute results
    results = camb.get_results(pars)

    # save parameter values to file
    f = open('%s/CAMB_recheck.params'%folder_ICs,'w');  f.write('%s'%pars);  f.close()

    # interpolate to get Pmm, Pcc...etc
    k, zs, Pkmm = results.get_matter_power_spectrum(minkh=2e-5, maxkh=kmax,
                                                    npoints=400, var1=7, var2=7,
                                                    have_power_spectra=True,
                                                    params=None)

    # do a loop over all redshifts
    for j,z in enumerate(zs):
        fout = '%s/Pk_m_z_recheck=%.3f.txt'%(folder_ICs,z)
        np.savetxt(fout, np.transpose([k,Pkmm[j,:]]))

    Num+=1

#/mnt/ceph/users/camels/Codes/2lpt/GLASS/dummy_glass_dmonly_64.dat
    ########################## write 2LPT parameter file #######################
    # parameter file for standard simulations
    a="""
Nmesh            1280
Nsample          640
Box              100000.0
FileBase         ics
OutputDir        ./
GlassFile        /mnt/ceph/users/lperez/ICsSAM/dummy_glass_dmonly_64.dat
GlassTileFac     10
Omega            %.4f
OmegaLambda      %.4f
OmegaBaryon      0.00
OmegaDM_2ndSpecies  0.0
HubbleParam      0.6711
Redshift         127
Sigma8           %.4f
SphereMode       0
WhichSpectrum    2
FileWithInputSpectrum   ./Pk_m_z_recheck=0.000.txt
InputSpectrum_UnitLength_in_cm  3.085678e24
ShapeGamma       0.201
PrimordialIndex  1.0

Phase_flip          0
RayleighSampling    0
Seed                %d

NumFilesWrittenInParallel 20
UnitLength_in_cm          3.085678e21
UnitMass_in_g             1.989e43
UnitVelocity_in_cm_per_s  1e5

WDM_On               0
WDM_Vtherm_On        0
WDM_PartMass_in_kev  10.0
    """%(Omega_m, 1.0-Omega_m, s8, i+5000)

    # save parameters to file
    f = open('%s/2LPT_recheck.param'%folder_ICs, 'w');  f.write(a);  f.close()
