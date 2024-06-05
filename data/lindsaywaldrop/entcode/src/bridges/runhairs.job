#!/bin/bash
#SBATCH -p defq
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --mail-type=FAIL

# echo commands to stdout
set -x 

# Set simulation number 
i=1

# Set number of hairs
a=25

# Set path 
WD="/home/waldrop@chapman.edu/entcode"

# move to working directory
cd "${WD}"/bin

# add appropriate modules
module load  openmpi-lw/4.1.3 hdf5/1.10.1 boost/1.60.0 silo/4.10.0 petsc/3.13.4-opt samrai/2.4.4-opt zlib-dev/1.2.11 libmesh/1.6.2-opt IBAMR/0.10.0-opt

# Copy data files to bin directory
cp "${WD}/data/input2d-files/input2d${i}" .
cp "${WD}/data/vertex-files/"${a}"hair_files/hairs${i}.vertex" .

# run MPI program
mpirun -n $SLURM_NTASKS ./main2d input2d${i}
rm input2d${i} hairs${i}.vertex

# check and zip finished simulation
cd "$WD"/results/ibamr/runs
if (test -d viz_IB2d${i}/lag_data.cycle_030000); then
    zip -r viz_IB2d${i}.zip viz_IB2d${i}/
else 
    echo "Simulation failed"
fi

