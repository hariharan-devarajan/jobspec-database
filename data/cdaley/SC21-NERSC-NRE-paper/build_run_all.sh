#!/bin/bash -l
#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -C dgx
#SBATCH -c 32
#SBATCH -G 1

echo -e "\n\nBuilding and running Babelstream..."
./BabelStream_build_script.sh

echo -e "\n\nBuilding and running BerkeleyGW (GPP)..."
./BerkeleyGW_build_script.sh

echo -e "\n\nBuilding and running Kokkos incremental tests..."
./Kokkos_build_script.sh

echo -e "\n\nBuilding and running the Kokkos version of TestSNAP with both CUDA and OpenMPTarget backends..."
./TestSNAP_build_script.sh

echo -e "\n\nBuilding and running the native OpenMP version of TestSNAP..."
./TestSNAP_native_build_script.sh
