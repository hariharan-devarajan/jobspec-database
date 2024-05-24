#!/bin/bash
#SBATCH -J 'CUT&Tag Pipeline' ### Job Name
#SBATCH --nodes=1 ### No. of Nodes
#SBATCH --ntasks-per-node 10 ### No. of Tasks
#SBATCH -o outLog ### Output Log File (Optional)
#SBATCH -e errLog ### Error Log File (Optional but suggest to have it)
#SBATCH --mail-user=some_email@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
 
snakemake --cores <num_cores> -s <snakefile_name>

# --cores must be the same as SBATCH --ntasks-per-node