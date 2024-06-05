#!/bin/bash
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=1
##SBATCH --time=01:00:00
##SBATCH --job-name=daskxgb
##SBATCH --mem=30G
##SBATCH --signal=B:SIGUSR1@120
##SBATCH --mail-type=end
##SBATCH --mail-type=fail
##SBATCH --mail-type=begin
##SBATCH -o dasktest.o.%J
##SBATCH -e dasktest.o.%J

scriptdir=""  # location for python scripts e.g. $HOME/dask_xgboost_slurm/scripts
data_dir=""
snps_dir=""
ids_dir=""

# setup python env
module load python_module  # replace with python/conda module
conda activate env_name  # replace with env name

python3 ${scriptdir}/subset_hdf5.py \
    --in_path "${data_dir}/infile.hdf5" \
    --out_path "${data_dir}/outfile.hdf5" \
    --ids "${ids_dir}/row_ids.txt" \
    --xkey 'x_adjusted' \
    --ykey 'y_adjusted'

