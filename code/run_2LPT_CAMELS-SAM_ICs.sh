#!/bin/bash

# make sure have: - module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
# run with: sh run_2LPT_CAMELS-SAM_ICs.sh
# if fails after “starting axes=3…” with a seg fault, its bc not enough memory/cpu i think! need to increase number of processes
N_cores=12 # need >8, <=12, empirically
echo "using N_cores=$N_cores"
for i in {0..0}
do
   echo "running 2LPT for realization $i"
   fn_2lpt="/cosmos_storage/kstoreyf/CAMELS-SAM_ICS/LH/LH_$i/2LPT_new.param"
   mpirun -np $N_cores ~/external/2lpt/2LPTic $fn_2lpt
done