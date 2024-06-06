#!/bin/bash

n_cores=(4 4 4 6 6 8 8 8 8 12 12)
n_mpiprocs=(4 2 1 6 1 8 4 2 1 6 2)

len=${#n_cores[@]}

# Create the folders
mkdir trials_GPU
for (( i=0; i<$len; i++ ))
do
  mkdir trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}
  touch trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}/md_meta${n_cores[i]}_${n_mpiprocs[i]}.pbs
done


# pbs files
for (( i=0; i<$len; i++ ))
do
  echo "#!/bin/bash\n#PBS -l select=1:ncpus=${n_cores[i]}:mpiprocs=${n_mpiprocs[i]}:ngpus=1:mem=1GB\n#PBS -l walltime=00:05:00
  \n#PBS -q short_gpuQ\n#PBS -N md_${n_cores[i]}_${n_mpiprocs[i]}\n#PBS -o md_${n_cores[i]}_${n_mpiprocs[i]}_out\n#PBS -e md_${n_cores[i]}_${n_mpiprocs[i]}_err
  \n\nmodule load gcc91\nmodule load openmpi-3.0.0\nmodule load BLAS\nmodule load gsl-2.5\nmodule load lapack-3.7.0
  \nmodule load cuda-11.3\n" > trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}/md_meta${n_cores[i]}_${n_mpiprocs[i]}.pbs

  awk 'NR==18' template.pbs >> trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}/md_meta${n_cores[i]}_${n_mpiprocs[i]}.pbs
  awk 'NR==20' template.pbs >> trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}/md_meta${n_cores[i]}_${n_mpiprocs[i]}.pbs
  awk 'NR==22' template.pbs >> trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}/md_meta${n_cores[i]}_${n_mpiprocs[i]}.pbs

  echo "\nexport OMP_NUM_THREADS=$(expr ${n_cores[i]} / ${n_mpiprocs[i]})
  \n/apps/openmpi-3.0.0/bin/mpirun -np ${n_mpiprocs[i]} /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s ../../md_meta.tpr -plumed meta.dat -nb gpu -pme auto" >> trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}/md_meta${n_cores[i]}_${n_mpiprocs[i]}.pbs
done


# Create the meta.dat files inside each subfolder
for (( i=0; i<$len; i++ ))
do
  cp meta.dat trials_GPU/trial_${n_cores[i]}_${n_mpiprocs[i]}/
done