#!/bin/bash
#SBATCH --job-name=Jul6_Cont_SS_HKNN_job    # Job name
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=clare.morris@alleninstitute.org     # Where to send mail  
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=500gb                     # Job memory request (per node)
#SBATCH --time=240:00:00               # Time limit hrs:min:sec
#SBATCH --output=Jul6_Cont_SS_HKNN_job_%j.log   # Standard output and error log
#SBATCH --partition celltypes         # Partition used for processing
#SBATCH --tmp=150G                     # Request the amount of space your jobs needs on /scratch/fast
mkdir $TMPDIR/tmp
singularity exec --bind=/scratch/fast/$SLURM_JOBID/tmp:/tmp docker://alleninst/mapping_on_hpc Rscript R_scripts/Cont_SS_HKNN.R > logfiles/Jul6_Cont_SS_HKNN_logfile
