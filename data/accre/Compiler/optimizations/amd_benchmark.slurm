#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --mem=2G
#SBATCH --output=results_amd.out
#SBATCH --constrain=amd

echo "   "
echo "Running on node: " $SLURM_NODELIST
echo "   "
echo "   "
echo "   "

echo "Running gcc_loop..."
echo " "
time ./gcc_loop
echo " "

echo "Running gcc_loop_O3..."
echo " "
time ./gcc_loop_O3
echo " "

echo "Running gcc_powern..."
echo " "
time ./gcc_powern
echo " "

echo "Running gcc_powern_O3..."
echo " "
time ./gcc_powern_O3
echo " "

echo "Running intel_loop..."
echo " "
time ./intel_loop
echo " "

echo "Running intel_loop_O3..."
echo " "
time ./intel_loop_O3
echo " "

echo "Running intel_powern..."
echo " "
time ./intel_powern
echo " "

echo "Running intel_powern_O3..."
echo " "
time ./intel_powern_O3
echo " "