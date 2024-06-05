#!/bin/bash
#SBATCH -n 1                                # number of comands issued in batch script at any times
#SBATCH --cpus-per-task=8                  # number of cores to allocate each task
#SBATCH -N 1                                # Request 1 node
#SBATCH --time=30:00:00

#SBATCH --gres=gpu:1                      # specify GPU count
#SBATCH -p sched_mit_ccoley                 # sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=9500                 # Request 4G of memory per CPU

#SBATCH -o logs/seed77777777_output_%x_%j.txt               # redirect output to output_JOBNAME_JOBID.txt
#SBATCH -e logs/seed77777777_error_%x_%j.txt                # redirect errors to error_JOBNAME_JOBID.txt
#SBATCH -J MEGAN_f10_pat4_stop8_maxn80                            # name of job
#SBATCH --mail-type=ALL               # Mail when job starts and ends
#SBATCH --mail-user=linmin001@e.ntu.edu.sg  # email address

##SBATCH --nodelist node1237
source /cm/shared/engaging/anaconda/2018.12/etc/profile.d/conda.sh
source env_seed77777777.sh

# this fixes RDKit problem
module load gcc/8.3.0

# preprocess training data from 3 csv files, only need to run once, DONE
# python3 bin/acquire.py uspto_50k

# get graphfeat, only need to run once, DONE, takes about 2 mins on 8 cores
# python3 bin/featurize.py uspto_50k megan_16_bfs_randat

# train, to run. Authors said 10 hours but it is way longer...about 20 hours on 1 GPU, bcos of the 16-epoch early-stopping patience.
python3 bin/train.py uspto_50k models/uspto_50k
# {
#     python3 bin/train.py uspto_50k models/uspto_50k
# }   ||  {
#     nvidia-smi
# }

# test
# python3 bin/eval.py models/uspto_50k --beam-size 50 --show-every 100