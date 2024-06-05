#!/bin/bash
#This is the launching script for Alpine3D. Please make sure the user section matches your needs!

########################## START  USER CONFIGURATION

#SBATCH --nodes=1                         # Number of requested nodes
#SBATCH --ntasks-per-node=24               # Number of threads per node
#SBATCH --account=ucb204_summit1
#SBATCH --time=04:00:00                   # Max wall time
#SBATCH --qos=normal                      # Specify testing QOS
#SBATCH --partition=shas                  # Specify Summit haswell nodes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eric.keenan@colorado.edu

# Settings
base_dir=$(pwd)
meteo_dir="/scratch/summit/erke2265/LISTON_EXPLORE/output/grids/"
src_dem_path=/pl/active/nasa_smb/Data/IS2_cycle_1_2_3_DEM_noFilter.tif

# Purge all modules and add only required ones
module purge
ml intel; ml proj; ml gdal; ml singularity/3.6.4; ml gnu_parallel

# Singularity settings
SINGULARITY_LOCALCACHEDIR=${base_dir}/../../
SINGULARITY_CACHEDIR=${base_dir}/../../
SINGULARITY_TMPDIR=${base_dir}/../../
export SINGULARITY_LOCALCACHEDIR
export SINGULARITY_CACHEDIR
export SINGULARITY_TMPDIR

# Purge all old output
rm -rf processed_output
rm -rf 1980* # Note that this line will need to be updated for time steps not in 1980!

# Write instructions (commands.txt)
rm -f commands.txt
for FILE in ${meteo_dir}/*.vw
do
	ts=$(basename -s .vw $FILE)
	echo "bash run.sh ${ts} ${src_dem_path} ${meteo_dir} ${base_dir}" >> commands.txt

done

# Execute computations
parallel --jobs ${SLURM_NTASKS} < commands.txt
