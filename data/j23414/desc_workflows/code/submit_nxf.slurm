#! /usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --job-name=Nextflow
#SBATCH --error=R_%x.%J.err
#SBATCH --output=R_%x.%J.out
# --mail-user=username@iastate.edu
# --mail-type=begin
# --mail-type=end

set -e
set -u

start=`date +%s`

# === Set working directory and in/out variables
cd ${SLURM_SUBMIT_DIR}
NEXTFLOW='/work/GIF/software/bin/nextflow'

# === Main Program (swap out the script here)
module load jdk
${NEXTFLOW} run script06.nf --outdir test_run

end=`date +%s`

# === Log msgs and resource use                          
scontrol show job ${SLURM_JOB_ID}
echo "ran submit_nxf.slurm: " `date` "; Execution time: " $((${end}-${start})) " seconds" >> LOGGER.txt
